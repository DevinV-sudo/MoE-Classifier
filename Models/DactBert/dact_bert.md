# DactBert Model Methods:  
---
### DactBert Class:
#### Purpose: 
This class is composed of the DACT-BERT model architecture. This architecture has additional modifications allowing
    it to take the pre-tuned weights from the BackBone class to use as initialization weights for joint training, as well
    as joint fine tuning each submodel component of the DACT-UNIT (accumulated output representation, accumulated confidence).
    Additionally there is added functionality in order for this architecture to be compatible with tuning the external training hyper-parameters.

    # Method Descriptions for `DactBert` Class

# Helper Functions:  
- Resolve_Config():  
    This helper function is one of the key players in this pipeline, it allows the model to accept the configuration dictionaries in any of the formats.
    This is valuable because it allows the model to generalize for tuning and training.

# Method Descriptions for DactBert Class

### __init__(self, classifier_config: dict = None, confidence_config: dict = None, backbone_weight_path = None, output_dimension: int = 3)

**Description:**

This is the constructor for the DactBert model. It initializes the model architecture, including the backbone (DistilBERT), classifier submodel, and confidence submodel. It also loads pretrained weights for the backbone model if provided, or loads the default weights if not.

**Arguments:**
- classifier_config (dict, optional): Configuration for the classifier submodel. If not provided, defaults will be used.
- confidence_config (dict, optional): Configuration for the confidence submodel. If not provided, defaults will be used.
- backbone_weight_path (str, optional): Path to the pretrained weights for the backbone model. If not provided, defaults are loaded.
- output_dimension (int, optional): The number of possible output classes. Defaults to 3.

---

### forward(self, input_ids, attention_mask)

**Description:**

This method performs the forward pass for the model. It processes the input through the DistilBERT backbone, computes the outputs for the classifier and confidence submodels, and determines the halting condition for early exit in inference mode. The method returns the model's output along with the halting sum and exit layers during training or inference.

**Arguments:**
- input_ids (Tensor): Input token IDs for the DistilBERT model.
- attention_mask (Tensor): Attention mask indicating which tokens to pay attention to (1 for real tokens, 0 for padding).

**Returns:**
- If training: A tuple containing the model output and the halting sum.
- If inference: A tuple containing the model output, the exit layers, and the halting sum.

---

### get_config(self, pretrained_weights_path: str=None)

**Description:**

This method generates a configuration dictionary containing all the necessary parameters to reconstruct the model. It includes paths to weights, configuration for the classifier and confidence submodels, and other model parameters.

**Arguments:**
- pretrained_weights_path (str, optional): Path to the pretrained weights. If provided, it will be added to the configuration.

**Returns:**
- A dictionary containing the model configuration, including paths to the backbone weights, classifier and confidence configurations, output dimension, and DistilBERT configuration.

---

### load_from_config(cls, config_file_path)

**Description:**

This is a class method that loads a model from a JSON configuration file. The configuration file must include the paths to pretrained weights for DistilBERT, configuration for the classifier and confidence submodels, and the output dimension. It will then instantiate the model, load the weights, and return the configured model.

**Arguments:**
- config_file_path (str): Path to the configuration file (JSON format) containing the model parameters.

**Returns:**
- The fully configured DactBert model, including pretrained weights if specified in the configuration file.

