#torch imports
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

#misc. imports
import logging
import os
import sys
import json

#import optuna
import optuna

#set up logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

#ensure required modules are on path
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if dataset_dir not in sys.path:
    sys.path.insert(0, models_dir)

#import back_bone weight generation modules
from BackBone.back_bone import BackBoneTune
from Models.DactBert.modular_submodel import ModularSubModel


#import the dataloaders
from Dataset.data_loaders import create_dataloaders

#Helper function to navigate different configurations through tuning and training pipeline
def resolve_config(config_input, default_config=None):
    '''
    Returns a Config Dictionary, 
    if config_input is None -> returns default config (if no default returns None)
    if config_input is a string (file_path) -> loads and returns the config dictionary
    if config_input is a dictionary -> returns the input dict
    '''

    #case where no input from optuna and now saved optimal config
    if config_input is None:
        logging.info(f"Loading Defaults...\n")
        if default_config is None:
            raise ValueError("No Config provided, and no default set.\n")
        return default_config

    #case where optimal config has been generated and exported
    elif isinstance(config_input, str):
        logging.info(f"Loading Config From Path...\n")
        if os.path.exists(config_input):
            with open(config_input, 'r')as f:
                config_dict = json.load(f)
            return config_dict
        else:
            raise ValueError(f"Config file path {config_input} does not exist.\n")

    #case where config is being tuned by optuna
    elif isinstance(config_input, dict):
        logging.info(f"Config is Dict Type.\n")
        return config_input
    else:
        raise TypeError("Config must be None, a file path string, or a dictionary.\n")



class DactBert(nn.Module):
    '''
    This class is composed of the DACT-BERT model architecture. This architecture has additional modifications allowing
    it to take the pre-tuned weights from the BackBone class to use as initialization weights for joint training, as well
    as joint fine tuning each submodel component of the DACT-UNIT (accumulated output representation, accumulated confidence).
    Additionally there is added functionality in order for this architecture to be compatible with tuning the external training hyper-parameters.

    Arguements:
        backbone_weight_path -> path to the pretuned weights for the model back_bone, if none, then loads a default
        classifier_config -> dict/path representing architecture parameters for confidence sub model
        confidence_config -> dict/path representing architecture parameters for confidence sub model
        tune_mode -> bool, if true then model outputs tuning metric, if False outputs standard outputs
    '''

    def __init__(self, 
                    classifier_config: dict = None,
                    confidence_config: dict = None,
                    backbone_weight_path = None,
                    output_dimension: int = 3
                    ):
        super(DactBert, self).__init__()

        #initialize general params
        self.output_dimension = output_dimension
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #set the default backbone weights
        default_weights_path = os.path.join(os.path.dirname(__file__), "DefaultWeights", "backbone_weights_tuned.bin")

        #check to see if back bone weights are provided, and if they are a valid path
        if not backbone_weight_path or not os.path.isfile(backbone_weight_path):
            logging.info(f"No BackBone weights provided, or path is in valid. Loading Defaults...\n")
            self.backbone_weight_path = default_weights_path

        #if they are provided set as instance attribute
        else:
            logging.info("Loading provided BackBone weights...\n")
            self.backbone_weight_path = backbone_weight_path
        
        #initialize the model backbone
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        config.output_hidden_states = True
        self.config = config
        self.model_back_bone = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

        #load in the backbone's weights
        pretrained_weights = torch.load(self.backbone_weight_path, map_location=self.device, weights_only=True)
        self.model_back_bone.load_state_dict(pretrained_weights)

        #initialize the hidden dimension
        self.hidden_dimension = self.config.dim
        
        #initialize Dact-submodel's Defaults
        classifier_default = {
            "num_layers": 1,
            "hidden_units": self.hidden_dimension // 2,
            "dropout": 0,
            "use_batch_norm": True,
            "activation": "ReLU"
        }
        confidence_default = {
            "num_layers": 1,
            "hidden_units": self.hidden_dimension // 2,
            "dropout": 0,
            "use_batch_norm": True,
            "activation": "ReLU"

        }
        #initialize DACT-submodels' configuration
        self.classifier_config = resolve_config(config_input=classifier_config, default_config=classifier_default)
        self.confidence_config = resolve_config(config_input=confidence_config, default_config=confidence_default)

        #initialize the DACT-unit's classifier submodel
        self.dact_output_classifier = ModularSubModel(is_classifier=True,
                                                    architecture_dict=self.classifier_config,
                                                    hidden_dimension = self.hidden_dimension,
                                                    output_dimension=self.output_dimension).get_sub_model()

        #Initialize the DACT-unit's confidence submodel
        self.dact_halting_confidence = ModularSubModel(is_classifier=False,
                                                        architecture_dict=self.confidence_config,
                                                        hidden_dimension=self.hidden_dimension,
                                                        output_dimension=self.output_dimension).get_sub_model()

    def forward(self, input_ids, attention_mask):
        #get the initial batchsize
        batch_size = input_ids.shape[0]

        #get the hidden states corresponding to each transformer layer
        outputs = self.model_back_bone(input_ids = input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        #the starting auxillary variables
        a = torch.zeros(batch_size, self.output_dimension, device=self.device)
        p = torch.ones(batch_size, device=self.device)

        #halting sum (accumulates for loss function)
        halting_sum = torch.zeros(batch_size, device=self.device)

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
            y_n = self.dact_output_classifier(cls_hs)

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

    def get_config(self, pretrained_weights_path: str=None):
        '''
        Return a configuration dictionary that conatins all the parameters nessary to reconstruct the model.
        '''
        logging.info(f"Saving model configuration\n")
        config_dict = {
            "backbone_weight_path": self.backbone_weight_path,
            "classifier_config": self.classifier_config,
            "confidence_config": self.confidence_config,
            "output_dimension": self.output_dimension,
            "distilbert_config": self.config.to_dict(),
            "pretrained_weights": None
        }

        #if pretrained weights are provided, then add the path
        if pretrained_weights_path:
            logging.info(f"Saving pretrained weights.\n")
            config_dict["pretrained_weights"] = pretrained_weights_path

        return config_dict

    @classmethod
    def load_from_config(cls, config_file_path):
        '''
        Class method to load a model from it's JSON configuration file.
        The configuration file should include:
            - backbone_weight_path -> path to pretrained weights of Distil-Bert
            - classifier_config -> configuration dict for DACT-submodel
            - confidence_config -> configuration dict for DACT-submodel
            - output_dimension -> integer (number of possible outputs)
            #Ideally an included pretrained weights file
        '''
        with open(config_file_path, "r") as f:
            config = json.load(f)

        #instiantiate model
        logging.info(f"Loading from Pre-built Configuration...\n")
        model = cls(
            backbone_weight_path=config["backbone_weight_path"],
            classifier_config=config["classifier_config"],
            confidence_config=config["confidence_config"],
            output_dimension=config["output_dimension"],
        )

        #check for pretrained weights
        if config["pretrained_weights"]:
            logging.info(f"Configuring Pre-Trained Weights...\n")
            #load in binary file
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrained_weights = torch.load(config["pretrained_weights"], map_location=device, weights_only=True)

            #initialize weights
            model.load_state_dict(pretrained_weights)

        #return the configured model
        logging.info(f"Model Ready.\n")
        return model

            

class TuningEngine():
    '''
    This class is responsible for tuning the submodel architecture of DactBert, the external training-parameters, and lambda-reg.
    It contains three methods, one for submodel tuning, one for hyper-parameter tuning and one for tuning lambda reg.
    (time out is in minutes unlike traditional optuna)
    '''
    def __init__(self, output_dimension: int = 3, time_out: int = None):
        #initialize the parameters
        self.output_dimension = output_dimension

        #convert to minutes
        if time_out:
            time_out = time_out*60
        self.time_out = time_out

        #initialize configuration elements to None
        self.classifier_config = None
        self.confidence_config = None
        self.external_param_config = None
        self.lambda_reg=None
    
    def submodel_objective(self, trial):
        #suggest hyperparameters for the classifier & package as a dictionary
        classifier_config = {
            "num_layers": trial.suggest_int("classifier_num_layers", 2, 6, step=1),
            "hidden_units": trial.suggest_int("classifier_hidden_units", 64, 768, step=64),
            "dropout": trial.suggest_float("classifier_dropout", 0.0, 0.5, step=0.1),
            "use_batch_norm": trial.suggest_categorical("classifier_use_batch_norm", [True, False]),
            "activation": trial.suggest_categorical("classifier_activation", ["ReLU", "LeakyReLU"]),
            "layer_strategy": trial.suggest_categorical("layer_strategy", ["pyramidal", "inverted_pyramidal", "bow_tie", "None"])
        }
        #suggest hyperparameters for the confidence & package as a dictionary
        confidence_config = {
            "num_layers": trial.suggest_int("confidence_num_layers", 2, 6, step=1),
            "hidden_units": trial.suggest_int("confidence_hidden_units", 64, 768, step=64),
            "dropout": trial.suggest_float("confidence_dropout", 0.0, 0.5, step=0.1),
            "use_batch_norm": trial.suggest_categorical("confidence_use_batch_norm", [True, False]),
            "activation": trial.suggest_categorical("confidence_activation", ["ReLU", "LeakyReLU"]),
            "layer_strategy": trial.suggest_categorical("layer_strategy", ["pyramidal", "inverted_pyramidal", "bow_tie", "None"])

        }

        #initialize an instance of DactBert (with both submodels -> modular)
        joint_tune_dact_bert = DactBert(classifier_config=classifier_config,
                                        confidence_config=confidence_config,
                                        output_dimension = self.output_dimension)

        #initialize training engine
        train_engine = TrainingEngine(model = joint_tune_dact_bert,
                                    external_param_config=None, #use default
                                    tune_mode=True, #output tuning metric
                                    lambda_reg=None, #use default
                                    save_model=False, #dont save
                                    save_metrics=False) #dont save

        #run the training loop, with pruning activated
        validation_loss, _, _ = train_engine.training_loop(trial=trial)

        #return the validation loss
        return validation_loss

    def joint_tune_study(self, run_tag: str = ''):
        '''
        This method is responsible for running architecture optimization over the DACT submodels.
        After the study has converged, the optimal architectures are saved as configuration files.
        (run-tag is for identifying studies later on)
        '''
        #define the study
        joint_study = optuna.create_study(sampler = optuna.samplers.RandomSampler(),
                                            direction="minimize",
                                            study_name="joint_architecture_study",
                                            load_if_exists=False)
        #optimize the objective
        if self.time_out:
            joint_study.optimize(self.submodel_objective, n_trials=25, timeout=self.time_out)
        else:
            joint_study.optimize(self.submodel_objective, n_trials=25)

        #create directory for the config files
        dact_unit_dir = "Models/DactBert/SubModelConfigs"
        os.makedirs(dact_unit_dir, exist_ok=True)

        #create subdirectories
        classifier_config_dir = os.path.join(dact_unit_dir, "classifier_configs")
        os.makedirs(classifier_config_dir, exist_ok=True)

        confidence_config_dir = os.path.join(dact_unit_dir, "confidence_configs")
        os.makedirs(confidence_config_dir, exist_ok=True)

        #generate save paths
        conf_config_path = os.path.join(confidence_config_dir, f"confidence_config_{run_tag}.json")
        class_config_path = os.path.join(classifier_config_dir, f"classifier_config_{run_tag}.json")

        #get configurations
        best_params = joint_study.best_trial.params

        #filter to generate configurations
        classifier_config = {
            key.replace("classifier_", ""): value 
            for key, value in best_params.items() 
            if key.startswith("classifier_")
        }
        confidence_config = {
            key.replace("confidence_", ""): value 
            for key, value in best_params.items() 
            if key.startswith("confidence_")
        }

        #save configs
        with open(conf_config_path, 'w') as conf_file:
            json.dump(confidence_config, conf_file, indent=4)
        with open(class_config_path, 'w') as class_file:
            json.dump(classifier_config, class_file, indent=4)

        # Load the configs back into the instance for later use
        with open(conf_config_path, 'r') as conf_file:
            self.confidence_config = json.load(conf_file)
        with open(class_config_path, 'r') as class_file:
            self.classifier_config = json.load(class_file)

        #return reloaded config dicts
        return self.classifier_config, self.confidence_config

    def train_tune_objective(self, trial):
        ''' This method handles the tuning of the external parameters or the training parameters '''

        #suggest training parameters
        external_param_config = {
            "split_prop": trial.suggest_categorical("split_prop", [0.1, 0.2, 0.3]),
            "batch_size": trial.suggest_int("batch_size", 32, 128, step=16),
            "reduction": trial.suggest_categorical("reduction", ["sum", "mean"]),
            "num_epochs": trial.suggest_int("num_epochs", 25, 75, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "weight_decay": trial.suggest_categorical("weight_decay", [0, 1e-5, 1e-4, 1e-3, 1e-2])
        }

        #initialize an instance of dact_bert
        train_tune_model = DactBert(classifier_config=self.classifier_config,
                        confidence_config=self.confidence_config,
                        output_dimension=self.output_dimension
                        )
        #initialize trainer instance
        train_engine = TrainingEngine(model=train_tune_model,
                                    external_param_config=external_param_config,
                                    tune_mode=True, #activate tune mode
                                    lambda_reg=None, #use default
                                    save_model=False, #dont save
                                    save_metrics=False) #dont save

        #run the training loop, with pruning activated
        validation_loss, _, _ = train_engine.training_loop(trial=trial)

        #return the validation loss
        return validation_loss

    def train_tune_study(self, run_tag: str=''):
        '''
        This method is responsible for tuning the training parameters. After the study has converged,
        optimal parameters are saved as a config file.
        (run-tag is for identifying studies later on)
        '''

        #initialize the objective
        train_study = optuna.create_study(sampler = optuna.samplers.RandomSampler(),
                                            direction="minimize",
                                            study_name="train_tune_study",
                                         load_if_exists=False)
        #run study                                 
        if self.time_out:
            train_study.optimize(self.train_tune_objective, n_trials=25, timeout=self.time_out)
        else:
            train_study.optimize(self.train_tune_objective, n_trials=25)

        #create directory for the config file
        external_config_dir = "Models/DactBert/ExternalConfigs"
        os.makedirs(external_config_dir, exist_ok=True)
        external_config_path = os.path.join(external_config_dir, f"external_config_{run_tag}.json")

        #get the best parameters
        best_params = train_study.best_trial.params

        #define the configuration
        external_config = best_params

        #save the configuration
        with open(external_config_path, 'w') as external_file:
            json.dump(external_config, external_file, indent=4)

        #reload the configuration as an instance attribute
        with open(external_config_path, 'r') as external_file:
            self.external_config = json.load(external_file)

        return self.external_config

    def lambda_objective(self, trial):
        ''' 
        This method handles the tuning of the regularization termL lambda_reg.
        lambda_reg controls the trade off between computational effeciency and model performance.
        To tune this feature, I will use optunas 'Dual Objective' capabilities with objectives being:
        average task loss, and a weighted average of exit layers
        '''

        #suggest lambda_reg values
        lambda_reg_trial = trial.suggest_float("lambda_reg", 5e-5, 5e-1, log=True)

        #initialize an instance of dact_bert
        lambda_tune_model = DactBert(classifier_config=self.classifier_config,
                                    confidence_config=self.confidence_config,
                                    output_dimension=self.output_dimension)

        #initialize trainer instance
        train_engine = TrainingEngine(model=lambda_tune_model,
                                    external_param_config=self.external_config,
                                    tune_mode=True,
                                    lambda_reg=lambda_reg_trial,
                                    save_model=False,
                                    save_metrics=False)
        #return dual objective
        _, avg_task_loss, avg_exit_layer = train_engine.training_loop(trial=trial)

        #return both tuning metrics
        return avg_task_loss, avg_exit_layer
        
    def lambda_reg_study(self, run_tag: str=''):
        '''
        This method is responsible for tuning lambda_reg, once the study converges,
        the final value is saved as a instance attribute. 
        '''

        #initialize objective
        lambda_study = optuna.create_study(sampler = optuna.samplers.RandomSampler(),
                                            directions=["minimize", "minimize"],
                                            study_name="lambda_tune_study",
                                            load_if_exists=False)
        #run the study
        if self.time_out:
            lambda_study.optimize(self.lambda_objective, n_trials=25, timeout=self.time_out)
        else:
            lambda_study.optimize(self.lambda_objective, n_trials=25)

        #get the best value of lambda_reg
        best_trial = min(lambda_study.best_trials, key=lambda t: t.values[1])
        best_lambda_value = best_trial.params["lambda_reg"]
        self.lambda_reg = best_lambda_value

        #return as an instance attribute
        return self.lambda_reg
    
         
    def final_train(self, run_tag: str = ''):
        '''
        This method checks to make sure all other tuning has been ran and then completes a final train,
        with all the optimal parameters and configurations. The trained model's full config file is pulled,
        and saved, and the model's weights are saved as well.
        '''

        #check to make sure all tuning is complete
        if not self.classifier_config or not self.confidence_config or not self.external_config or not self.lambda_reg:
            raise ValueError("All tuning must be compeleted to run 'final_train'.\n")

        #initialize the model
        final_model = DactBert( classifier_config=self.classifier_config,
                                confidence_config=self.confidence_config,
                                output_dimension=self.output_dimension)
        #initialize the trainer
        train_engine = TrainingEngine(model=final_model,
                                    external_param_config=self.external_config,
                                    tune_mode=False,
                                    lambda_reg=self.lambda_reg,
                                    save_model=True,
                                    save_metrics=True,
                                    run_tag=run_tag)
        #train the final model
        metrics_path, model_path = train_engine.training_loop()

        #return the paths
        return metrics_path, model_path

    def run_pipeline(self, run_tag: str = ''):
        '''
        Runs the complete tuning pipeline in the following order:
         1. Submodel tuning
         2. External parameter tuning
         3. Lambda regularization tuning
         4. Final training and saving of the model
        Returns the paths to the saved metrics and model.
        '''
        try:
            logging.info(f"Starting Sub-model Tuning...\n")
            self.joint_tune_study(run_tag=run_tag)
        except Exception as e:
            logging.error(f"Failed to tune sub-model. Failed with {e}\n")
            return None

        try:
            logging.info(f"Starting External Parameter Tuning...\n")
            self.train_tune_study(run_tag=run_tag)
        except Exception as e:
            logging.error(f"Failed to tune external parameters. Failed with {e}\n")
            return None

        try:
            logging.info(f"Starting Lambda-Reg Tuning...\n")
            self.lambda_reg_study(run_tag=run_tag)
        except Exception as e:
            logging.error(f"Failed to tune lambda-reg. Failed with {e}\n")

        try:
            logging.info(f"Starting Final Train...\n")
            metrics_save_path, model_weights_path = self.final_train(run_tag=run_tag)
            logging.info(f"Saved Model Config at: {model_weights_path}\n")
            return metrics_save_path, model_weights_path
        except Excpetion as e:
            logging.error(f"Failed to complete Dact-Bert Tuning. Failed with: {e}\n")

class TrainingEngine():
    '''
    This class handles the training of 'DactBert'. It has dual functionality in the sense
    that it both trains an instance of 'DactBert' but also is equipped to operate as a tuning loop
    for tuning both the submodel architecture and the external hyper-parameters.
    '''
    def __init__(self, model: DactBert,
                external_param_config=None, 
                tune_mode: bool = False, 
                lambda_reg: float = None, 
                save_model: bool = False, 
                save_metrics: bool = False,
                run_tag: str = '',
                train_dataloader = None,
                val_dataloader = None
                ):

        #set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #initialize the model and send to device
        self.model = model.to(self.device)

        #initialize lambda_reg with default if not provided
        if not lambda_reg:
            self.lambda_reg = 5e-2
        else:
            self.lambda_reg = lambda_reg

        #initialize flags
        self.tune_mode = tune_mode
        self.save_model = save_model
        self.run_tag = run_tag
        self.save_metrics = save_metrics

        #set the defaults for external parameters
        default_external_params = {
            "split_prop": 0.3,
            "batch_size": 128,
            "reduction": "sum",
            "num_epochs": 25,
            "learning_rate": 1e-4,
            "weight_decay": 0,
        }
        #initialize and resolve the external parameters
        self.external_param_config = resolve_config(config_input=external_param_config, default_config=default_external_params)

        #create dataloaders from config if not provided
        if not train_dataloader or not val_dataloader:
            logging.info("No loaders provided loading from config...\n")
            batch_size=self.external_param_config["batch_size"]
            split_prop=self.external_param_config["split_prop"]
            self.train_dataloader, self.val_dataloader, self.test_dataloader = create_dataloaders(batch_size=batch_size, split_prop=split_prop)
            logging.info(f"Successfully generated loaders. Number of batches: {len(self.train_dataloader)}\n")

        else:
            #initialize dataloaders if provided
            logging.info(f"Initializing provided dataloaders...\n")
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            logging.info(f"Successfully generated dataloaders.\n")

        #initialize criterion
        reduction = self.external_param_config["reduction"]
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

        #initialize optimizer
        weight_decay = self.external_param_config["weight_decay"]
        learning_rate = self.external_param_config["learning_rate"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = learning_rate, weight_decay=weight_decay)

        #Initialize num-epochs
        self.num_epochs = self.external_param_config["num_epochs"]

    def train_step(self):
        ''' Runs one Epoch of training, outputs epoch loss epoch accuracy '''
        #set model to training mode
        logging.info(f"Model in training mode...\n")
        self.model.train()

        #storage for training metrics
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        #iterate through batches
        for batch in self.train_dataloader:
            #send batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            #unpack the batch data
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"].long()

            #set gradients to zero
            self.optimizer.zero_grad()

            #complete a forward pass through the model
            outputs, halting_sum = self.model(input_ids=input_ids, attention_mask=attention_mask)

            #compute the task loss (cross-entropy between CLS rep. and labels)
            task_loss = self.criterion(outputs, labels)

            #regularization loss based off of accumulated halting confidence
            reg_loss = self.lambda_reg * halting_sum.mean()

            #compute the total loss
            loss = task_loss + reg_loss

            #complete a backward pass
            loss.backward()
            self.optimizer.step()

            #accumulate loss
            train_loss += loss.item()

            #compute the model's predictions
            preds = outputs.argmax(dim=1)

            #accumulate predictions and true labels
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
            

        #concatentate model's training metrics across all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        #compute the training accuracy
        train_accuracy = (all_preds == all_labels).float().mean().item()

        #compute the average per-batch loss
        train_loss = train_loss / len(self.train_dataloader)

        #return results
        results = [train_loss, train_accuracy]
        return results

    def validation_step(self):
        ''' This method is responsible for running one epoch worth of validation'''
        #set the model to eval mode
        self.model.eval()

        #storage for validation metrics
        all_preds = []
        all_labels = []
        exited_layers = []
        validation_loss = 0.0
        val_task_loss = 0.0

        #turn off gradients
        with torch.no_grad():
            for batch in self.val_dataloader:
                #send batch items to device
                batch = {k:v.to(self.device) for k, v in batch.items()}

                #unpack the data
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"].long()

                #gather outputs from model
                outputs, exit_layers, halting_sum = self.model(input_ids, attention_mask)

                #get the validation task loss
                task_loss = self.criterion(outputs, labels)

                #store the task loss
                val_task_loss += task_loss.item()

                #get the validation regularization loss
                reg_loss = self.lambda_reg * halting_sum.mean()

                #get the combined loss
                loss = task_loss + reg_loss
                validation_loss += loss.item()

                #get batch predictions
                preds = outputs.argmax(dim=1)

                #store the metrics
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())
                exited_layers.append(exit_layers)

            #accumulate batch metrics and compute the validation accuracy
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            validation_accuracy = (all_preds==all_labels).float().mean().item()

            #compute the epoch validation loss
            validation_loss = validation_loss / len(self.val_dataloader)

            #compute the epoch task loss (for tuning)
            val_task_loss = val_task_loss / len(self.val_dataloader)

            #compute an element wise sum over the exit layers
            aggregated_exit_layers = np.sum(np.array(exited_layers), axis=0)
            
            #return results
            results = [validation_loss, validation_accuracy, aggregated_exit_layers, val_task_loss]
            return results

    def training_loop(self, trial=None):
        ''' 
        This methods acts as both the training loop and the tuning loop. If the model is in tune mode,
        then the loop returns just the best validation loss for tuning, it also reports intermediate values
        if trial is set and the model is tuning mode for pruning.
        '''
        if self.tune_mode:
            logging.info("Model in tuning mode - returning only tune metric.\n")
        else:
            logging.info(f"Model in training mode - Save Metrics: {self.save_metrics}\n")

        #initialize storage for metrics
        train_losses = []
        train_accuracies = []
        validation_losses = []
        validation_accuracies = []
        validation_exits = []
        validation_task_losses = 0.0

        #initialization for functional early stopping
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        
        if '__file__' in globals():
            base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            base_dir = os.getcwd()

        #define storage directories
        model_metrics_dir = os.path.join(base_dir, "Metrics")
        model_weights_dir = os.path.join(base_dir, "Weights")
        model_architecture_directory = os.path.join(base_dir, "Configuration")

        #create if they don't exist
        os.makedirs(model_metrics_dir, exist_ok=True)
        os.makedirs(model_weights_dir, exist_ok=True)
        os.makedirs(model_architecture_directory, exist_ok=True)

        #define save paths
        metrics_file = f"dact_bert_metrics_{self.run_tag}.json"
        metrics_save_path = os.path.join(model_metrics_dir, metrics_file)

        weights_file = f"dact_bert_weights_{self.run_tag}.bin"
        model_weights_save_path = os.path.join(model_weights_dir, weights_file)

        model_arch_file = f"dact_bert_configuration_{self.run_tag}.json"
        model_arch_save_path = os.path.join(model_architecture_directory, model_arch_file)
        
        
        #start training / tuning loop
        logging.info(f"Beginning training...\n")
        for epoch in range(1, self.num_epochs + 1):
            logging.info(f"Epoch: {epoch}/{self.num_epochs}\n")

            #perform one training step
            train_loss, train_accuracy = self.train_step()

            #store results
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            #perform one validation step
            validation_loss, validation_accuracy, aggregated_exit_layers, validation_task_loss = self.validation_step()

            #store results
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)
            validation_task_losses += validation_task_loss
            validation_exits.append(aggregated_exit_layers)

            #if in tuning mode, report intermediate value
            if self.tune_mode is True and trial is not None:

                #check to make sure trial is not multi-objective
                if len(trial.study.directions) == 1:
                    trial.report(validation_loss, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned(f"Trial pruned at epoch: {epoch}")

            #if the current validation loss is lower than the historical one
            if validation_loss < best_val_loss:
                #reset patience counter
                patience_counter = 0

                #update the best validation loss
                best_val_loss = validation_loss

                #if weights directory has been initialized save
                if self.save_metrics:
                    logging.info(f"Saving model weights to {model_weights_dir}...\n")
                    torch.save(self.model.state_dict(), model_weights_save_path)
                     
            #if current is not better accumulate counter
            patience_counter += 1

            #log per epoch metrics
            logging.info(f"Metrics for Epoch: {epoch}\n")
            logging.info(f"Training Loss: {train_loss:.8f}, Training Accuracy: {train_accuracy:.8f}\n")
            logging.info(f"validation Loss: {validation_loss:.8f}, Validation Accuracy: {validation_accuracy:.8f}\n")

            # Check if early stopping criteria met
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch} epochs with no validation loss improvement.\n")
                break

        #log end of training and best loss
        logging.info(f"Training Complete. Best Validation Loss: {best_val_loss}\n")

        #get the total exit layer counts & total samples
        exit_layer_counts = np.sum(np.array(validation_exits), axis=0)

        #calculate a weighted average exit layer
        layers = np.arange(len(exit_layer_counts))

        avg_exit_layer = np.sum(layers * exit_layer_counts) / np.sum(exit_layer_counts)
        logging.info(f"average exit_layer {avg_exit_layer}")

        #calculate the average task loss over epochs
        avg_task_loss = validation_task_losses / self.num_epochs

        #save model metrics
        history = {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "train_accuracies": train_accuracies,
            "validation_accuracies": validation_accuracies,
            "aggregated_exit_layers": aggregated_exit_layers.tolist()
        }

        #compile and save results:
        results = {}

        #save metrics if activated
        if self.save_metrics:
            logging.info(f"Saving model to metric directory: {model_metrics_dir}\n")
            with open(metrics_save_path, 'w') as metrics_file:
                json.dump(history, metrics_file, indent=4)
            results["metrics_path"] = metrics_save_path
        else:
            results["metrics_path"] = None

        #save model config if activated
        if self.save_model:
            logging.info(f"Saving model config to architecture directory: {model_architecture_directory}\n")
            config_dict = self.model.get_config(pretrained_weights_path = model_weights_save_path)
            with open(model_arch_save_path, "w") as config_file:
                json.dump(config_dict, config_file, indent = 4)
            results["model_config_path"] = model_arch_save_path
        else:
            results["model_config_path"] = None

        #return outputs based on tuning mode
        if not self.tune_mode:
            logging.info(f"Training Complete, returning data paths...\n")
            return results
        else:
            logging.info(f"Training Complete, returning tune-metric..\n")
            return best_val_loss, avg_task_loss, avg_exit_layer


def main():
    '''
    This is an example of tuning the model, I am tuning all in one go however
    you can also tune each component seperately.
    '''

    #initialize tuning engine
    tuning_engine = TuningEngine(output_dimension=3, time_out=60)

    #run the tuning pipeline
    _,_ = tuning_engine.run_pipeline(run_tag="TUNED")

    

if __name__ == "__main__":
    main()








     