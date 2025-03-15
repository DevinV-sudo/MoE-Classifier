#torch imports
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

#transformer imports
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

#default imports
import logging
import os
import sys
import json
import numpy as np

#tuning imports
import optuna
from optuna.exceptions import TrialPruned

#ensure required modules are on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#import the dataloaders module
from Dataset.data_loaders import create_dataloaders

class BackBone(nn.Module):
    '''
    This acts as the back bone for DACT-Bert, i.e. the transformer model from which the hidden states are
    exited from. In this class Distil-Bert is trained with a single linear output layer to predict the output
    course given the input course. The weights solely from the back bone are then extracted for the initialization
    weights in the DACT-Bert model. The entire DACT-Bert model is then trained jointly using these weights as a starting point.
    '''

    def __init__(self, output_dimension: int = 3):
        super(BackBone, self).__init__()

        #load in the base model (Distil-Bert)
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        self.backbone = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

        #output dimensionality
        self.output_dimension = output_dimension

        #get the hidden dimension
        self.hidden_dimension = config.dim

        #initialize the temporary output classifier
        self.temp_model_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dimension, self.output_dimension)
        )

    def forward(self, input_ids, attention_mask):
        #get the initial batchsize
        batch_size = input_ids.shape[0]

        #get the CLS token from the last hidden state of the model
        cls_token = self.backbone(input_ids = input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        #pass the CLS token to the temporary model head
        output = self.temp_model_head(cls_token)

        #return the output
        return output

class BackBoneTrain():
    ''' This class handles the training of the BackBone model, as well as the extraction and exportation of weights.'''
    def __init__(self, back_bone: BackBone,
                        num_epochs: int = 25,
                        lr: float = 5e-4,
                        batch_size: int = 128,
                        split_prop: float = 0.3,
                        is_tune: bool = False,
                        run_tag: str = ''):

        #initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #initialize dataloaders
        self.train_dataloader, self.validation_dataloader, _ = create_dataloaders(batch_size = batch_size, split_prop=split_prop)

        #initialize the 'BackBone' model
        self.back_bone_model = back_bone.to(self.device)

        #initialize num_epochs
        self.num_epochs = num_epochs

        #initialize loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.back_bone_model.parameters(), lr=lr)

        #initialize learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(self.train_dataloader),
            epochs=self.num_epochs,
            pct_start=0.1
        )

        #initialize tuning flag
        self.is_tune = is_tune

        #initialize tag
        self.run_tag = run_tag

    def backbone_train_step(self):
        ''' Orchestrates a training step for the backbone model '''

        #set model into training mode
        logging.info(f"Entering Training State...\n")
        self.back_bone_model.train()

        #storage for epoch wise loss
        all_preds = []
        all_labels = []
        epoch_loss = 0.0

        #initialize mixed precision
        scaler = torch.amp.GradScaler()

        #iterate through the batches
        for batch in self.train_dataloader:
            #send batch to device if device exists
            batch = {k: v.to(self.device) for k, v in batch.items()}
            #unpack the data
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"].long()

            #set gradients to zero
            self.optimizer.zero_grad()

            #enable AMP (Automatic Mixed Precision)
            with torch.amp.autocast():
                #get outputs
                outputs = self.back_bone_model(input_ids=input_ids, attention_mask=attention_mask)

                #calculate loss
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()

            #backward pass
            scaler.scale(loss).backward()

            #unscale
            scaler.unscale_(self.optimizer)

            #gradient clipping
            torch.nn.utils.clip_grad_norm_(self.back_bone_model.parameters(), max_norm=1.0)

            #step the optimizer and the scheduler
            scaler.step(self.optimizer)
            self.scheduler.step()

            #update
            scaler.update()

            #accumulate predictions, probabilities and true labels
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

        #concatenate results from all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        #compute the accuracy
        epoch_accuracy = (all_preds == all_labels).float().mean().item()

        #compute average per batch loss
        epoch_loss = epoch_loss / len(self.train_dataloader)    

        #return the accuracy and loss
        return (epoch_loss, epoch_accuracy)

    def backbone_val_step(self):
        ''' Orchestrates a validation step for the back bone model '''

        #set the model into eval mode
        logging.info(f"Entering Evaluation State...\n")
        self.back_bone_model.eval()

        #storage for the predictions and labels
        all_preds = []
        all_labels = []
        val_losses = 0.0

        #dont update gradients
        with torch.no_grad():
            for batch in self.validation_dataloader:
                #send batch items to device
                batch = {k:v.to(self.device) for k, v in batch.items()}

                #unpack the data
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"].long()

                #gather the output representation
                outputs = self.back_bone_model(input_ids=input_ids, attention_mask=attention_mask)

                #calculate loss
                loss = self.criterion(outputs, labels)
                val_losses += loss.item()

                #get the predictions
                preds = outputs.argmax(dim=1)
                
                #store the predictions
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())

        #compute validation accuracy
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        accuracy = (all_preds==all_labels).float().mean().item()

        #validation loss
        validation_loss = val_losses / len(self.validation_dataloader)

        #return results
        return (validation_loss, accuracy)

    def backbone_train_loop(self, trial=None):
        '''
        This function runs a training loop for the back bone model,
        If tune=True, and the trial is passed, then report intermediate values
        back to optuna for pruning
        '''

        #storage for model metrics
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        val_losses = []

        #tracking the best validation loss
        best_val_loss = float("inf")
        patience = 2
        patience_counter = 0
        
        if not self.is_tune:
            #weights save path
            best_model_dir = "Models/BackBone/backbone_weights"
            os.makedirs(best_model_dir, exist_ok=True)
            weights_file = f"backbone_weights_{self.run_tag}.bin"
            best_model_path = os.path.join(best_model_dir, weights_file)

            #metrics save path
            metrics_dir = "Models/BackBone/backbone_train_results"
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_file = f"backbone_metrics_{self.run_tag}.json"
            metrics_save_path = os.path.join(metrics_dir, metrics_file)

        #interate through num_epochs
        logging.info(f"Beginning training...\n")
        for epoch in range(1, self.num_epochs + 1):
            logging.info(f"Epoch: {epoch}/{self.num_epochs}\n")

            #perform one step of training
            train_loss, train_accuracy = self.backbone_train_step()

            #store results
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            #perform one step of validation
            val_loss, val_accuracy = self.backbone_val_step()

            #store results
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            #report intermediate value
            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned(f"Trial Pruned at epoch: {epoch}")

            #save model weights if curr val loss is better than historical best
            if val_loss < best_val_loss:
                #reset patience counter
                patience_counter = 0

                #update 'best validation loss'
                best_val_loss = val_loss

                #in tuning mode - no need to save weights
                if not self.is_tune:
                    #save the weights just from the 'distil-bert' backbone.
                    torch.save(self.back_bone_model.backbone.state_dict(), best_model_path)
                    logging.info(f"New best validation loss: {best_val_loss:.4f}\nWeights saved at: {best_model_path}\n")
            
            else:
                #update patience
                patience_counter += 1
            
            #log per epoch metrics
            logging.info(f"Metrics for Epoch: {epoch}\n")
            logging.info(f"Training Loss: {train_loss:.8f}, Training Accuracy: {train_accuracy:.8f}\n")
            logging.info(f"validation Loss: {val_loss:.8f}, Validation Accuracy: {val_accuracy:.8f}\n")
            
            # Check if early stopping criteria met
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch} epochs with no validation loss improvement.\n")
                break
        #log end of training and best loss
        logging.info(f"Training Complete. Best Validation Loss: {best_val_loss}\n")

        #save the model results
        history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }

        #if not in tuning mode return and save results:
        if not self.is_tune:
            logging.info(f"Saving model's results...\n")
            with open(metrics_save_path, "w") as file:
                json.dump(history, file, indent=4)
            
            #return the weights path and metrics path
            return best_model_path, metrics_save_path
        
        #if in tuning mode return the best_val_loss
        else:
            logging.info(f"In Tune mode: Best Val Loss: {best_val_loss}\n")
            return best_val_loss

class BackBoneTune():
    ''' This class tunes the training params for the backbone-train class'''
    def __init__(self, output_dimension):
        
        #initialize output dimension
        self.output_dimension = output_dimension 
    
    def objective(self, trial):
        #suggest hyper parameters
        lr = trial.suggest_float("lr", 5e-6, 1e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        num_epochs = trial.suggest_int("num_epochs", 20, 30)
        split_prop = trial.suggest_categorical("split_prop", [0.2, 0.3])

        #instiantiate back-bone model
        back_bone_model = BackBone(output_dimension=self.output_dimension)

        #create trainer (is_tune = True)
        back_bone_trainer = BackBoneTrain(
                                back_bone=back_bone_model,
                                num_epochs=num_epochs,
                                lr=lr,
                                batch_size=batch_size,
                                split_prop=split_prop,
                                is_tune=True)
        
        #run the training steps, instead of the loop to apply a 'pruner'
        best_val_loss = back_bone_trainer.backbone_train_loop(trial=trial)

        #return the best validation loss
        return best_val_loss

    def run_study(self):
        ''' Runs the study based on the above objective function '''
        study = optuna.create_study(sampler = optuna.samplers.RandomSampler(),
                                    direction="minimize",
                                    study_name="back_bone_tuner",
                                    load_if_exists=False)

        study.optimize(self.objective, n_trials=20)

        #create save directory for best hyper-parameters
        param_dir = "Models/BackBone/backbone_train_params"
        os.makedirs(param_dir, exist_ok=True)
        param_path = os.path.join(param_dir, "backbone_hyperparams.json")

        #get best hyper params and best val_loss
        best_params = study.best_trial.params
        best_loss = study.best_trial.value
        logging.info(f"Best Validation Loss: {best_loss:.2f}\nBest Parameters:\n{best_params}\n")

        #save the hyper parameters
        logging.info(f"Saving Back-Bone Parameters to:\n {param_path}\n")
        with open(param_path, "w") as params_file:
            json.dump(best_params, params_file, indent=4)

        #return the save path
        return param_path

    def train_tuned_backbone(self):
        ''' 
        This method runs a study, loads the tuned parameters, and trains a back bone. 
        The final pretrained weights are exported
        '''

        #initialize and run the study
        parameter_path = self.run_study()

        #load in parameters
        with open(parameter_path, 'r') as file:
            hyper_parameters = json.load(file)

        #initialize back bone
        back_bone_model = BackBone(output_dimension=self.output_dimension)

        #initialize trainer (is_tune = False)
        back_bone_trainer = BackBoneTrain(
                            back_bone=back_bone_model,
                            num_epochs=hyper_parameters["num_epochs"],
                            lr=hyper_parameters["lr"],
                            batch_size=hyper_parameters["batch_size"],
                            split_prop=hyper_parameters["split_prop"],
                            is_tune=False,
                            run_tag="tuned"
        )

        #run training
        best_model_path, metrics_save_path = back_bone_trainer.backbone_train_loop()
        return (best_model_path, metrics_save_path)



####################################
#----------EXAMPLE-USAGE-----------#
####################################

if __name__ == "__main__":
    ''' 
    Here I am going to tune the model-backbone weights for use in the Dact-Bert Model.
    '''

    #initialize tuner
    tuner = BackBoneTune(output_dimension=3)
    
    #run tuning pipeline
    model_weights_path, model_metrics_path = tuner.train_tuned_backbone()
    
    




        
