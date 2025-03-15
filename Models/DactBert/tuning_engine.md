### Overview:  
The tuning of this model happens in stages, first the two architectures of the sub models of the Dact Unit are tuned, then the hyperparameters of the entire of the model are tuned.
then finally the the lambda reg regularization is tuned using a dual objective trial to minimize the loss while minimizing the number of transformer layers used.

## Method Descriptions for TuningEngine Class
---

### __init__(self, output_dimension: int = 3, time_out: int = None)

**Description:**

This is the constructor for the TuningEngine class. It initializes the engine with the output dimension and timeout parameters. The timeout is converted from minutes to seconds if provided.

**Arguments:**
- output_dimension (int, optional): The number of output classes. Defaults to 3.
- time_out (int, optional): Timeout value for the tuning process, in minutes. It is converted to seconds.

---

### submodel_objective(self, trial)

**Description:**

This method defines the objective for tuning the submodels (classifier and confidence). It suggests hyperparameters for both submodels using Optuna trials, and then runs a training loop to evaluate the validation loss.

**Arguments:**
- trial: An Optuna trial object that suggests hyperparameters.

**Returns:**
- validation_loss: The validation loss after training.

---

### joint_tune_study(self, run_tag: str = '')

**Description:**

This method runs the architecture optimization study for the DACT submodels. After optimization, the best architecture configurations are saved to disk.

**Arguments:**
- run_tag (str, optional): A tag for identifying studies later on.

**Returns:**
- classifier_config: The optimized classifier configuration.
- confidence_config: The optimized confidence configuration.

---

### train_tune_objective(self, trial)

**Description:**

This method defines the objective for tuning external training parameters. It suggests training parameters such as batch size, learning rate, and number of epochs, and then runs the training loop to evaluate the validation loss.

**Arguments:**
- trial: An Optuna trial object that suggests training parameters.

**Returns:**
- validation_loss: The validation loss after training.

---

### train_tune_study(self, run_tag: str='')

**Description:**

This method runs the study for tuning external training parameters. Once the study converges, the best parameters are saved to disk.

**Arguments:**
- run_tag (str, optional): A tag for identifying studies later on.

**Returns:**
- external_config: The optimized external training configuration.

---

### lambda_objective(self, trial)

**Description:**

This method defines the objective for tuning the regularization term (lambda_reg). It uses dual objectives—task loss and exit layer performance—and suggests a lambda_reg value to balance computational efficiency and model performance.

**Arguments:**
- trial: An Optuna trial object that suggests lambda_reg values.

**Returns:**
- avg_task_loss: The average task loss after training.
- avg_exit_layer: The weighted average of exit layer performance.

---

### lambda_reg_study(self, run_tag: str='')

**Description:**

This method runs the study for tuning the lambda_reg parameter. Once the study converges, the best lambda_reg value is saved as an instance attribute.

**Arguments:**
- run_tag (str, optional): A tag for identifying studies later on.

**Returns:**
- lambda_reg: The best lambda_reg value after tuning.

---

### final_train(self, run_tag: str = '')

**Description:**

This method runs the final training step using all the optimized parameters and configurations. The trained model's config file and weights are saved.

**Arguments:**
- run_tag (str, optional): A tag for identifying the final training run.

**Returns:**
- metrics_path: The path to the saved training metrics.
- model_path: The path to the saved model weights.

---

### run_pipeline(self, run_tag: str = '')

**Description:**

This method runs the complete tuning pipeline in the following order:
1. Submodel tuning
2. External parameter tuning
3. Lambda regularization tuning
4. Final training and saving of the model

It returns the paths to the saved metrics and model.

**Arguments:**
- run_tag (str, optional): A tag for identifying the tuning run.

**Returns:**
- metrics_save_path: The path to the saved training metrics.
- model_weights_path: The path to the saved model weights.
