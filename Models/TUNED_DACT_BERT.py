#torch imports
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

#ensure required modules are on path
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#import the modular submodel function
from Models.MODULAR_DACT_BERT import build_sub_model

class TUNED_DACT_BERT(nn.Module):
    '''
    The same architecture as DACT_BERT but in this case instead of the modular iterations,
    where each sub model is either initialised with default values or from a pretrained dict,
    and the other's architecture is tuned, each sub model is built using the build sub-model
    function and the hyper parameters for the entire system are tuned.
    '''
    def __init__(self, classifer_dict: dict, confidence_dict: dict, output_dimension: int=3):
        super(TUNED_DACT_BERT, self).__init__()

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
        self.hidden_size = config.dim

        #initalize the output dimension
        self.output_dimension = output_dimension

        #initialise the output classifier with the pre-tuned dictionary
        self.dact_output_classifer = build_sub_model(
            hidden_size = self.hidden_size,
                output_dim= self.output_dimension,
                num_layers=classifer_dict["num_layers"],
                hidden_units=classifer_dict["hidden_units"],
                dropout=classifer_dict["dropout"],
                use_batch_norm=classifer_dict["use_batch_norm"],
                final_activation=nn.Softmax(dim=1)
            )
        
        #initialize the confidence predictor using the pre-tuned dictionary
        self.dact_halting_confidence = build_sub_model(
                    hidden_size=self.hidden_size,
                    output_dim=1,
                    num_layers=confidence_dict["num_layers"],
                    hidden_units=confidence_dict["hidden_units"],
                    dropout=confidence_dict["dropout"],
                    use_batch_norm=confidence_dict["use_batch_norm"],
                    final_activation=nn.Sigmoid()
                )
    
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
        