#torch imports
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

class VanillaBert(nn.Module):
    '''
    This class contains a 'vanilla' implementation of Distil-Bert, for comparison against the tuned DACT-BERT model.
    The body consists of the 'frozen' distil-bert architecture, and the classifying head is the same as the classifying head which creates
    accumulated representations of the CLS token in DACT-BERT.
    '''
    def __init__(self, output_dimension: int = 3):
        super(VanillaBert, self).__init__()
        
        #configure distil bert backbone to output all hidden states
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        self.config = config
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

        #turning off the gradients for distil bert
        for param in self.bert.parameters():
            param.requires_grad = False

        #hidden dimensionality
        hidden_size = config.dim

        #output dimensionality
        self.output_dimension = output_dimension

        #defining the vanilla output classifier head
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dimension),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, attention_mask):
        ''' forward pass through vanilla-bert's architecture'''

        #get the outputs from pretrained distil-bert (the CLS token from the last hidden state)
        cls_token = self.bert(input_ids = input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        #pass the CLS token to the classifer
        output = self.classifier_head(cls_token)

        #return the output
        return output


        