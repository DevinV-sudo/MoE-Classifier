### TrainingEngine Class Overview

The TrainingEngine class handles both the training and tuning of the DactBert model. It supports hyperparameter optimization (e.g., through Optuna) or standard training, along with features like model saving, early stopping, and metric tracking.

### Key Methods:
- **Training Loop**: Runs a training step for a single epoch.
- **Validation Loop**: Runs a validation step for a single epoch.
- **Hyperparameter Tuning**: Supports hyperparameter optimization (e.g., Optuna).
- **Early Stopping**: Stops training if no improvement in validation loss is seen for a set number of epochs.
- **Model and Metric Saving**: Saves model weights and training metrics during training.

---

### __init__(self, model: DactBert, ...)

The constructor initializes the TrainingEngine with various configurations and parameters for training.

### Parameters:
- **model**: The DactBert model to be trained.
- **external_param_config**: Configuration parameters such as learning rate, batch size, etc.
- **tune_mode**: Flag to enable hyperparameter tuning.
- **lambda_reg**: Regularization strength for halting confidence.
- **save_model**: Flag to save model weights.
- **save_metrics**: Flag to save training metrics.
- **run_tag**: Tag for identifying the run.
- **train_dataloader**: Optional custom training dataloader.
- **val_dataloader**: Optional custom validation dataloader.

### Attributes:
- self.model: The model to be trained.
- self.optimizer: AdamW optimizer.
- self.criterion: CrossEntropy loss function.
- self.external_param_config: Configuration parameters like batch size, learning rate, etc.
- self.train_dataloader, self.val_dataloader, self.test_dataloader: Dataloaders for training, validation, and testing.

---

## train_step(self)

This method performs a single epoch of training.

### Functionality:
- Sets the model to training mode.
- Iterates over the training dataloader.
- Computes task loss (cross-entropy) and regularization loss based on halting confidence.
- Updates the model weights using backpropagation.
- Returns the average training loss and accuracy.

---

## validation_step(self)

This method performs a single epoch of validation.

### Functionality:
- Sets the model to evaluation mode.
- Iterates over the validation dataloader.
- Computes the task loss, regularization loss, and returns aggregated metrics like validation loss, accuracy, and exit layers.
- Disables gradients to save memory.

---

## training_loop(self, trial=None)

The main loop for training and tuning. It loops through the epochs and manages early stopping, model saving, and hyperparameter tuning.

### Functionality:
- Loops through the epochs, calling train_step and validation_step.
- Optionally reports intermediate results for hyperparameter tuning (e.g., using Optuna).
- Implements early stopping if no improvement in validation loss is seen.
- Saves model weights, metrics, and configurations.
- Returns the final results, including paths to saved metrics and model weights, or the best validation loss in tuning mode.

### Early Stopping:
- Stops training if no validation improvement is seen for a set number of epochs (patience).

### Hyperparameter Tuning:
- Supports tuning using trial, which reports validation loss at each epoch for hyperparameter optimization.

---

## Key Losses:
- **Task Loss**: Cross-entropy loss for classification.
- **Regularization Loss**: Based on the halting confidence, encouraging the model to make decisions earlier.

### Saving:
- **Model Weights**: The model's weights are saved if self.save_model is True.
- **Metrics**: Training and validation metrics are saved if self.save_metrics is True.
