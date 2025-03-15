# Back-Bone Methods:
---  
#### Purpose:  
The BackBone class defines a simple version of Distil-Bert, which is trained and finetuned in order to extract it's weights (From the Transformer Block) to use as pretrained weights for the DACT-BERT model, providing a good starting point for further tuning and training. 
By leveraging these pretrained weights, the model benefits from improved performance when trained as a whole.  

### Classes & Methods:

#### BackBone:
---  
**Overview**:  
This class defines the backbone of the DACT-BERT model, which is a DistilBERT model with an additional linear layer for pretraining. The trained backbone weights are later used to initialize the full DACT-BERT model.  

**Methods:**  
- __init__(self, output_dimension: int=3)  
  Initializes the BackBone model with DistilBERT and a linear classifier head.

- forward(self, input_ids, attention_mask)  
  Passes input tokens through the DistilBERT model and extracts the CLS token, which is then passed through a linear layer for output prediction.

#### BackBoneTrain:  
---
**Overview:**
Handles training of the BackBone model, manages data loading, training steps, validation, and saving the pretrained weights.

**Methods:**
- __init__(self, back_bone, num_epochs, lr, batch_size, split_prop, is_tune, run_tag):  
Initializes the training setup, including data loaders, loss function, optimizer, and learning rate scheduler.

- backbone_train_step(self):  
Runs a single training step, computing loss and accuracy while updating model weights.  

- backbone_val_step(self):  
Runs a single validation step, computing validation loss and accuracy.  

- backbone_train_loop(self, trial=None):  
Executes multiple epochs of training and validation, implements early stopping, and saves the best model weights.  

#### BackBoneTune:  
---
**Overview:**  
This class tunes hyperparameters for the BackBoneTrain class using Optuna and runs the final model training with the best-found parameters.  

**Methods:**
- __init__(self, output_dimension):  
Initializes the tuning class with the specified output dimension.  

- objective(self, trial):  
Defines the objective function for Optuna, selecting the best learning rate, batch size, number of epochs, and split proportion.  

- run_study(self):  
Runs the Optuna study to find the best hyperparameters.  

- train_tuned_backbone(self):  
Uses the best hyperparameters from the study to train the final BackBone model and export pretrained weights.  


