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

class ModularSubModel():
    '''
    This class orchestrates the construction of a sub-model as a component of the DACT unit in the DACT-BERT architecture.
    It takes a dictionary containing model architecture parameters, when tuning this dictionary represents the suggested
    parameters from optuna, when tuning is completed the dictionary will be the optimal architecture, and additionally,
    this class contains two defaults, one for each submodel in DACT-BERT
    '''
    def __init__(self, is_classifier: bool, architecture_dict: dict, hidden_dimension: int, output_dimension: int = 3):

        #initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #initialize arguements
        self.is_classifier = is_classifier
        self.architecture_dict = architecture_dict
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension

        #initialize activation dictionary:
        self.active_dict = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.ReLU()
        }

    def build_sub_model(self):
        '''
        This method is responsible for creating either type of DACT-UNIT submodel from a provided
        architecture dict.
        '''
        try:
            #construct the sub-model architecture
            num_layers = self.architecture_dict["num_layers"]
            hidden_units = self.architecture_dict["hidden_units"]
            dropout = self.architecture_dict["dropout"]
            use_batch_norm = self.architecture_dict["use_batch_norm"]
            activation_key = self.architecture_dict["activation"]
            activation = self.active_dict[activation_key]

            #determine final activation & output based on sub-model type
            if self.is_classifier:
                logging.info(f"Building classifier submodel from dict...\n")
                output_dimension = self.output_dimension
                final_activation = nn.Softmax(dim=1)
            else:
                logging.info(f"Building confidence submodel from dict...\n")
                output_dimension = 1
                final_activation = nn.Sigmoid()
                
            #storage for layer accumulation
            layers = []

            #initial input features (updates through loop)
            in_features = self.hidden_dimension

            #iterate through the number of layers
            for i in range(num_layers):
                layers.append(nn.Linear(in_features, hidden_units))
                layers.append(activation)
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_units))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_features = hidden_units

            #append the output layer (and specified final activation)
            layers.append(nn.Linear(in_features, output_dimension))
            if final_activation:
                layers.append(final_activation)
            
            #combine into a sequential object
            logging.info(f"Returning custom architecture.\n")
            self.submodel = nn.Sequential(*layers)

            return self.submodel

        except Exception as e:
            logging.error(f"Submodel construction failed. Failed with: {e}\n")
            return None

    def get_sub_model(self):
        ''' 
        This method is responsible for either delegating to the build_sub_model method
        in the case of a provided architecture dict, or returning the requested default model.
        '''
        
        try:
            #determine whether to produce a default, or a specified architecture
            if self.architecture_dict:
                logging.info(f"Architecture dict provided.\n")
                self.submodel = self.build_sub_model()
            else:
                logging.info(f"No architecture dict provided\n")
                if self.is_classifier:
                    logging.info(f"Returning default classifier.\n")
                    self.submodel = self.default_classifier
                else:
                    logging.info(f"Returning default confidence submodel.\n")
                    self.submodel = self.default_confidence
            
            #return submodel
            return self.submodel
            
        except Exception as e:
            logging.error(f"Failed to retreive submodel. Failed with {e}\n")
            return None




        