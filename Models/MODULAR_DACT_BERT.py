#torch imports
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

#building a customizable sub-model
def build_sub_model(hidden_size, output_dim, num_layers, hidden_units, dropout, use_batch_norm, final_activation=None):
    '''
    This acts as a modular sub model for both the classifier head portion of the DACT unit,
    as well as the simple MLP for calculating the accumulated exit confidence.
    depending on the usage of this modular sub model (either classifying or confidence) the output layer will be fixed.
    '''

    #for accumulating the layers
    layers = []
    in_features = hidden_size

    #iterate through the number of layers
    for i in range(num_layers):
        layers.append(nn.Linear(in_features, hidden_units))
        layers.append(nn.ReLU())
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_units))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_features = hidden_units

    #append the output layer (and specified final activation)
    layers.append(nn.Linear(in_features, output_dim))
    if final_activation:
        layers.append(final_activation)
    
    #combine into a sequential object
    return nn.Sequential(*layers)

class ModularDACT_BERT(nn.Module):
    '''
    DACT-BERT architecture, with modular DACT sub units.
    '''
    def __init__(self, output_dimension: int=3, num_layers: int = 3,
                hidden_units: int = 128, dropout: float = 0.1,
                use_batch_norm: bool = False, which_node: str = "classifier", pretrained_dict = None):

        super(ModularDACT_BERT, self).__init__()
        
        #check for which modular node
        options = ["classifier", "confidence"]

        #verify node selected is valid
        if which_node not in options:
            logging.error(f"Node: {which_node} is not a valid selection.\n")
            logging.info(f"Valid options: {options}\n")
            return None

        #configure DACT-BERT
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        config.output_hidden_states = True
        self.config = config
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

        #freeze the BERT layers
        for param in self.bert.parameters():
            param.requires_grad = False
        
        #turn the bert segment to eval mode
        self.bert.eval()
        
        #initialize the hidden size
        hidden_size = config.dim

        #initalize the output dimension
        self.output_dimension = output_dimension

        #initialize the modular output classifier (if selected)
        if which_node == "classifier":
            self.dact_output_classifer = build_sub_model(
                hidden_size = hidden_size,
                output_dim=output_dimension,
                num_layers=num_layers,
                hidden_units=hidden_units,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                final_activation=nn.Softmax(dim=1)
            )

            #check if there is a pretrained dictionary
            if pretrained_dict != None:
                #if there is a pretrained dictionary then use it to build the sub model
                self.dact_halting_confidence = build_sub_model(
                    hidden_size=hidden_size,
                    output_dim=1,
                    num_layers=pretrained_dict["num_layers"],
                    hidden_units=pretrained_dict["hidden_units"],
                    dropout=pretrained_dict["dropout"],
                    use_batch_norm=pretrained_dict["use_batch_norm"],
                    final_activation=nn.Sigmoid()
                )
            else:
                #use the default confidence sub model
                self.dact_halting_confidence = nn.Sequential(
                    #takes the hidden state from the transformer layer
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1),
                    nn.Sigmoid())

        #initialize the modular confidence sub-model (if selected)
        else:
            self.dact_halting_confidence = build_sub_model(
                hidden_size=hidden_size,
                output_dim=1,
                num_layers=num_layers,
                hidden_units=hidden_units,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                final_activation=nn.Sigmoid()
            )

            #check if there is a pretrained dictionary
            if pretrained_dict != None:
                self.dact_output_classifer = build_sub_model(
                    hidden_size=hidden_size,
                    output_dim=output_dimension,
                    num_layers=pretrained_dict["num_layers"],
                    hidden_units=pretrained_dict["hidden_units"],
                    dropout=pretrained_dict["dropout"],
                    use_batch_norm=pretrained_dict["use_batch_norm"],
                    final_activation=nn.Softmax(dim=1)
                )

            else:
                #This unit computes the intermediate output (using base sub model)
                self.dact_output_classifer = nn.Sequential(
                    #take the hiddent state from the transformer layer
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, output_dimension),
                    nn.Softmax(dim=1))

    def forward(self, input_ids, attention_mask):
        #pull the batch size from the input
        batch_size = input_ids.shape[0]

        #get the hidden states corresponding to each transformer layer
        outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        a = torch.zeros(batch_size, self.output_dimension, device=input_ids.device)
        p = torch.ones(batch_size, device=input_ids.device)

        #halting sum (accumulates for loss function)
        halting_sum = torch.zeros(batch_size, device=input_ids.device)

        #total steps for the stopping condition
        total_steps = len(hidden_states) -1

        #storage for transformer exit layers
        exit_layers = np.zeros(len(hidden_states)-1)

        #flag for marking exits
        exited = False

        #iterate through hidden states
        for idx, hs in enumerate(hidden_states[1:]):
            #extract the [CLS] token for both DACT units
            cls_hs = hs[:, 0, :]

            #get the intermediate outputs:
            y_n = self.dact_output_classifer(cls_hs)

            #get the intermediate confidence
            h_n = self.dact_halting_confidence(cls_hs).squeeze(-1)

            #update inital value of a (combined outputs)
            a = (y_n * p.unsqueeze(1)) + (a * (1 - p.unsqueeze(1)))
            
            #accumulate the halting sum
            halting_sum = halting_sum + h_n

            #update intial value of p (confidence)
            p = p * h_n

            #if the model is not training (inference) check early exit
            if not self.training:
                #d represents the remaining steps
                d = total_steps -1

                #for each sample get the highest probability, and the second (runner-up)
                top_probs, _ = a.max(dim=1)
                sorted_probs, _ = a.sort(dim=1, descending=True)
                second_probs = sorted_probs[:, 1] # runner up

                #the halting condition
                condition = (top_probs * ((1-p) ** d)) >= (second_probs + (p ** d))

                #if the halting condition holds for every sample in the batch
                if condition.all():
                    #mark the transformer layer exited
                    exit_layers[idx] += 1

                    #set the flag to true
                    exited = True
                    break
        #if the model is in inference mode, and no exits were made mark the final exit
        if not self.training and not exited:
            exit_layers[-1] += 1
            
        #if the model is training return both the estimated representation (a) and the halting sum
        output = a
        if self.training:
            return output, halting_sum
        
        #return the final accumulate CLS token as well as the layers where exited if in "inference mode"
        return output, exit_layers, halting_sum