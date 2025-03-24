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
            base_hidden_units = self.architecture_dict["hidden_units"]
            dropout = self.architecture_dict["dropout"]
            use_batch_norm = self.architecture_dict["use_batch_norm"]
            activation_key = self.architecture_dict["activation"]
            activation = self.active_dict[activation_key]
            layer_strategy = self.architecture_dict["layer_strategy"]

            #determine final activation & output based on sub-model type
            if self.is_classifier:
                logging.info(f"Building classifier submodel from dict...\n")
                output_dimension = self.output_dimension
                final_activation = nn.Softmax(dim=1)
            else:
                logging.info(f"Building confidence submodel from dict...\n")
                output_dimension = 1
                final_activation = nn.Sigmoid()
            
            #create model filter (exactly the same across modular sub models)
            filter_layer = nn.Linear(self.hidden_dimension, self.hidden_dimension)
            filter_activation =activation

            #storage for layer accumulation
            layers = []

            #append the first two static input layers
            layers.append(filter_layer)
            layers.append(filter_activation)

            #non static input layer, changes by base hidden units, activation, dropout, norm etc. prepares layer architecture dimensionality
            input_layer = nn.Linear(self.hidden_dimension, base_hidden_units)
            layers.append(input_layer)

            #if batch norm add it
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(base_hidden_units))

            #add the activation
            layers.append(activation)

            #dropout if specified
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            #initial input features (updates through loop)
            in_features = base_hidden_units

            #saving previous hidden units for bowtie architecture
            previous_hidden = []

            #initializing the reverse indexing
            reverse_idx = -1

            #if the number of layers specified is != 1, run construction loop, else just input and out put layers
            if num_layers > 1:
                logging.info(f" Number of layers specfied is greater than one, running construction loop.\n")
                #iterate through the number of layers
                for i in range(1, num_layers+1):
                    #based on strategy change hidden units (first layer is fully connected)
                    if layer_strategy == "pyramidal":
                        if i == 1:
                            logging.info("Structure is Pyramidal.\n")

                        #gradually decrease hidden units
                        hidden_units = max(int(base_hidden_units * (1 - i/num_layers)), 32)
                        logging.info(f"layer: {i}, units: {hidden_units}\n")

                    elif layer_strategy == "inverted_pyramidal":
                        if i == 1:
                            logging.info(f"Structure is Inverted Pyramidal.\n")

                        #gradually increase layer sizes
                        hidden_units = int(base_hidden_units * (1 + i/num_layers))
                        logging.info(f"layer: {i}, units: {hidden_units}\n")

                    elif layer_strategy == "bow_tie":
                        #first half decrease hidden units
                        if i == 1:
                            logging.info(f"Structure is BowTie.\n")

                        if i <= num_layers // 2:
                            logging.info(f"First half of Bowtie (Decrement)\n")
                            hidden_units = int(base_hidden_units * (1 - i/num_layers))
                            #store the hidden units for back stepping
                            previous_hidden.append(hidden_units)
                            logging.info(f"layer: {i}, units: {hidden_units}\n")

                        #second half increase hidden units (back step through previous hidden units)
                        else:
                            logging.info(f"Second half of Bowtie (Increment)\n")
                            #initialize as most recent hidden unit counr
                            hidden_units = previous_hidden[reverse_idx]
                            logging.info(f"layer: {i}, units: {hidden_units}\n")

                            #decrement by one
                            reverse_idx  = reverse_idx -1

                    else:
                        #if not strategy keep base units consistent
                        logging.info(f"No structure specified. Consistent hidden units applied.\n")
                        hidden_units = base_hidden_units

                    #store the layer & activation
                    layers.append(nn.Linear(in_features, hidden_units))

                    #if batch norm is involved add it
                    if use_batch_norm:
                        layers.append(nn.BatchNorm1d(hidden_units))

                    #add the activation (post norm if specified)
                    layers.append(activation)

                    #dropout if specified
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    
                    #reset the in features
                    in_features = hidden_units

                #if the layer structure is bow tie, append the final bowtie (missed by the for loop)
                if layer_strategy == "bow_tie":
                    logging.info(f"Adding final increment layer.\n")
                    layers.append(nn.Linear(in_features, base_hidden_units))
                    logging.info(f"layer: Final, units: {base_hidden_units}\n")

                    #if batch norm is involved add it
                    if use_batch_norm:
                        layers.append(nn.BatchNorm1d(base_hidden_units))

                    #add the activation (post norm if specified)
                    layers.append(activation)

                    #dropout if specified
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))

                    #reset the in_features, for final activation
                    in_features = base_hidden_units
            
            #append the output layer (and specified final activation)
            layers.append(nn.Linear(in_features, output_dimension))
            if final_activation:
                layers.append(final_activation)
            
            #combine into a sequential object
            logging.info(f"Returning custom architecture.\n")
            self.submodel = nn.Sequential(*layers)

            #return sequential unit
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


########EXAMPLE USAGE##########
pyramidal_test_dict = {
                    "num_layers": 1,
                    "hidden_units": 128,
                    "dropout": 0.3,
                    "use_batch_norm": True,
                    "activation": "ReLU",
                    "layer_strategy": "pyramidal"
}

inverted_pyramidal_test_dict = {
                    "num_layers": 1,
                    "hidden_units": 128,
                    "dropout": 0.3,
                    "use_batch_norm": True,
                    "activation": "ReLU",
                    "layer_strategy": "inverted_pyramidal"
}

bow_tie_test_dict = {
                    "num_layers": 1,
                    "hidden_units": 128,
                    "dropout": 0.3,
                    "use_batch_norm": True,
                    "activation": "ReLU",
                    "layer_strategy": "bow_tie"
}

#other params for testing
output_dimension = 3
hidden_dimension = 768
arch_list = [pyramidal_test_dict, inverted_pyramidal_test_dict, bow_tie_test_dict]

def main():
    logging.info(f"Starting Modular Architecture Testing...\n")
    for arch in arch_list:
        #run the classifier sub model
        architecture = arch["layer_strategy"]
        logging.info(f"Displaying Modular Classifier Architecture for {architecture}\n")
        model = ModularSubModel(is_classifier=True, architecture_dict=arch, hidden_dimension=hidden_dimension, output_dimension=output_dimension)
        classifier_sub_model = model.get_sub_model()

        #print the classifier sub model architecture
        print(f"{classifier_sub_model}\n")

        #run the confidence sub model
        logging.info(f"Displaying Modular Confidence Architecture for {architecture}\n")
        model = ModularSubModel(is_classifier=False, architecture_dict=arch, hidden_dimension=hidden_dimension, output_dimension=output_dimension)
        confidence_sub_model = model.get_sub_model()

        #print the confidence sub model architecture
        print(f"{confidence_sub_model}\n")
        print(f"{"--" * 50}\n")

if __name__ == "__main__":
    main()




        