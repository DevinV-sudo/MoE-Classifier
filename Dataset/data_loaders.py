#imports
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import load_from_disk

#torch imports
import torch
from torch.utils.data import DataLoader

#transformer imports
from transformers import DistilBertTokenizer
from transformers import DataCollatorWithPadding


#misc. imports
import os
import logging

#set up logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def load_datasets(base_path, split):
    #valid split options
    options = ["Train", "Valid", "Test"]

    #check split choice
    logging.info(f"Checking Split Choice: {split}...")
    if split not in options:
        raise ValueError(f"{split} is not an option, options: {options}\n")
    logging.info(f"Selected valid split.\n")
    
    #check if base_path exists:
    if not os.path.isdir(base_path):
        logging.error(f"Base path does not exist\n")
        return None
    
    #check if split choice exists
    dataset_name = f"{split}Dataset"
    dataset_path = os.path.join(base_path, dataset_name)
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset: {dataset_name} does not exist at: {dataset_path}\n")
        return None

    #if both the base path exists and the file, load the dataset
    try:
        logging.info(f"Attempting to load data\n")
        data_set = load_from_disk(dataset_path)
        logging.info(f"Successfully Loaded Dataset.\n")
        return data_set
    
    except Exception as e:
        logging.error(f"Failed to load data: {e}\n")
        return None

def load_tokenizer(model_name):
    #simple function just loads the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    return tokenizer

def create_dataloaders(batch_size, base_path="/projects/dsci410_510/SyntheticQueryData/", model_name='distilbert-base-uncased', split_prop=0.2):
    '''
    This function loads in the dataset for each split, and generates dataloaders for each
    each data loader is initialized, and returned for use in model training
    '''
    #ensure that selected split option is valid
    split_proportions = {}

    #extracting proportion data was split on from directory name pattern
    split_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    for d in split_dirs:
        try:
            split_str = d.split("_DataSplitProp")[0]
            split_float = float(split_str.replace("_", "."))
            split_proportions[split_float] = d
        except ValueError:
            pass
    
    if split_prop not in split_proportions.keys():
        logging.error("User Selected Invalid Split.\n")
        logging.info(f"Valid Splits: {split_proportions.keys()}\n")
        return None

    #load in the tokenizer
    try:
        logging.info(f"Loading tokenizer...\n")
        tokenizer = load_tokenizer(model_name)
        logging.info(f"Successfully loaded tokenizer.\n")

    except Exception as e:
        logging.error(f"Failed to load tokenizer, failed with: {e}\n")
        return None

    #load in each dataset
    try:
        base_path = os.path.join(base_path, split_proportions[split_prop])
        logging.info(f"Loading Datasets...\n")
        train_dataset = load_datasets(base_path, "Train")
        valid_dataset = load_datasets(base_path, "Valid")
        test_dataset = load_datasets(base_path, "Test")
        logging.info(f"Successfully loaded datasets.\n")
    
    except Exception as e:
        logging.error(f"Failed to load datasets, failed with: {e}\n")
        return None

    #initialize the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #load dataloaders
    try:
        logging.info(f"Creating data loaders...\n")
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator, drop_last=True)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator, drop_last=True)
        logging.info(f"Successfully created data loaders.\n")

        #return dataloaders
        loaders = [train_dataloader, valid_dataloader, test_dataloader]
        return loaders

    except Exception as e:
        logging.error(f"Failed to generate data loaders, failed with: {e}\n")
        return None

#example usage
if __name__ == "__main__":
    #generate loaders
    loaders = create_dataloaders(base_path="/projects/dsci410_510/SyntheticQueryData", model_name='distilbert-base-uncased', split_prop = 0.2, batch_size=64)

    train_dataloader, valid_dataloader, test_dataloader = loaders

    #print example from each loader:
    for name, dataloader in zip(["Train", "Validation", "Test"], [train_dataloader, valid_dataloader, test_dataloader]):
        example_batch = next(iter(dataloader))  # Get the first batch
        print(f"{name} DataLoader Example:")
        print(example_batch)
        print("-" * 50)

    