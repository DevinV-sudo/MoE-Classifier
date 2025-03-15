#torch imports
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

#misc. imports
import logging

#set up logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class DACT_BERT(nn.Module):
    '''
    This model is approximated from the DACT-BERT model proposed by 
    Cristóbal Eyzaguirre1, Felipe del Río1, Vladimir Araujo1, and Alvaro Soto
    in their paper DACT-BERT: Differentiable Adaptive Computation Time for an Efficient BERT Inference.
    '''
    def __init__(self, output_dimension: int = 3, weights_path = None):
        super(DACT_BERT, self).__init__()
        
        #configure distil bert backbone to output all hidden states
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        config.output_hidden_states = True
        self.config = config
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

        #initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #load in the initial weights (if available)
        if weights_path != None:
            logging.info(f"Loading in pretrained back-bone weights...\n")
            
            #load in the file using 'torch' (binary file)
            pretrained_weights = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.bert.load_state_dict(pretrained_weights)
            
        else:
            logging.info(f"No provided weights, using defaults.\n")

        #hidden dimensionality
        hidden_size = config.dim

        #output dimensionality
        self.output_dimension = output_dimension

        #This unit computes the intermediate output 
        self.dact_output_classifer = nn.Sequential(
            #take the hiddent state from the transformer layer
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dimension),
            nn.Softmax(dim=1))

        #this unit computes the "confidence"
        self.dact_halting_confidence = nn.Sequential(
            #takes the hidden state from the transformer layer
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid())

        
    def forward(self, input_ids, attention_mask):
        #get the initial batchsize
        batch_size = input_ids.shape[0]

        #get the hidden states corresponding to each transformer layer
        outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        #the starting auxillary variables
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