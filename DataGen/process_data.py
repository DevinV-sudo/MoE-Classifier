#imports
import pandas as pd
import torch
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from datasets import Dataset
from transformers import DistilBertTokenizer

#set up logging config
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def read_data():
    ''' Read in data from "generate_data'''
    path = "DataGen/data/question_data"
    if os.path.isdir(path):
        file_path = os.path.join(path, "QuestionData.csv")
        data = pd.read_csv(file_path)
        
        #shuffle the data
        data = data.sample(n=data.shape[0], random_state=42, replace = False)
        return data
    else:
        logging.error(f"File path: {path} does not exist.")
        return None
    
def data_split(data):
    #split target data
    input = data["Question"]
    output = data["Label"]

    #log action
    logging.info(f"Splitting data....\n")

    #split into test and temp, temp will be used for generating the validation set, equal class proportions
    temp_X, X_test, temp_y, y_test = train_test_split(input, output, test_size=0.1, random_state=42, stratify = output)

    #split temp into train and val sets
    X_train, X_val, y_train, y_val = train_test_split(temp_X, temp_y, test_size=0.1, random_state=42, stratify = temp_y)

    #successfully split data
    logging.info(f"Successfully split data:\n")

    #log size
    logging.info(f"Train size: {X_train.shape[0]}\n")
    logging.info(f"Val size: {X_val.shape[0]}\n")
    logging.info(f"Test size: {X_test.shape[0]}\n")

    #ensure string type
    X_train = X_train.astype(str)
    X_val = X_val.astype(str)
    X_test = X_test.astype(str)

    #recombine into dataframes
    train_data = pd.concat([X_train, y_train], axis = 1)
    valid_data = pd.concat([X_val, y_val], axis = 1)
    test_data = pd.concat([X_test, y_test], axis = 1)

    #save the splits
    logging.info(f"Saving splits...\n")

    #directory for dataframes
    split_dir = "DataGen/data/data_splits"
    os.makedirs(split_dir, exist_ok=True)
    
    #save the training data:
    train_path = os.path.join(split_dir, "train_data.csv")
    train_data.to_csv(train_path, index=False)

    #save the validation data:
    valid_path = os.path.join(split_dir, "valid_data.csv")
    valid_data.to_csv(valid_path, index=False)

    #save the test data:
    test_path = os.path.join(split_dir, "test_data.csv")
    test_data.to_csv(test_path, index=False)
    
    paths = [train_path, valid_path, test_path]
    y_splits = [y_train, y_val, y_test]
    
    data = (paths, y_splits)
    return data

#loading in the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_input(data):
    return tokenizer(data["Question"], truncation = True, max_length=250)

def encode_tensor_dataset(splits, save_dir):
    ''' 
    This function will take in the data splits, and
    one hot encode their target variables as well as embed the 
    input variables and store them as input-output pairs 
    '''
    #unpack the data splits
    logging.info(f"Unpacking data...\n")
    paths, y_splits = splits
    y_train, y_val, y_test = y_splits
    train_path, valid_path, test_path = paths
    
    #reshape y variables
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    y_val = np.array(y_val).reshape(-1, 1)

    #initialize encoder
    encoder = OneHotEncoder(sparse_output=False)
    
    #fit encoder only on train and transform the rest
    logging.info(f"Transforming y-splits....\n")
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)
    y_test_encoded = encoder.transform(y_test)
    logging.info(f"Successfully transformed the y-splits\n")

    #load in the datasets
    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    test_data = pd.read_csv(test_path)

    #ensure that strings are intact
    train_data["Question"] = train_data["Question"].astype("str")
    valid_data["Question"] = valid_data["Question"].astype("str")
    test_data["Question"] = test_data["Question"].astype("str")

    #convert to hugging face dataset
    try:
        logging.info(f"Attempting to generate hugging face datasets...\n")
        train_dataset = Dataset.from_pandas(train_data)
        valid_dataset = Dataset.from_pandas(valid_data)
        test_dataset = Dataset.from_pandas(test_data)
        logging.info(f"Successfully generated hugging face datasets.\n")

    except Exception as e:
        logging.error(f"Failed to generate hugging face datasets: {e}\n")
        return None

    #map the X datasets and remove the unused columns
    try:
        logging.info(f"Attempting to map tokenizer to hugging face data...\n")
        train_dataset = train_dataset.map(tokenize_input, batched=True).remove_columns(["Question", "Label"])
        valid_dataset = valid_dataset.map(tokenize_input, batched=True).remove_columns(["Question", "Label"])
        test_dataset = test_dataset.map(tokenize_input, batched=True).remove_columns(["Question", "Label"])

        #add back the one-hot encoded y_columns
        logging.info(f"Adding back encoded columns...\n")
        train_dataset = train_dataset.add_column("labels", y_train_encoded.tolist())
        valid_dataset = valid_dataset.add_column("labels",  y_val_encoded.tolist())
        test_dataset = test_dataset.add_column("labels", y_test_encoded.tolist())
        logging.info(f"Successfully created datasets.\n")
    
    except Exception as e:
        logging.error(f"Failed to generate hugging face datasets.\n")
        logging.error(f"Failed with error: {e}\n")
        return None


    #save the hugging face datasets
    try:
        logging.info(f"Attempting to save hugging face datasets to disk...\n")
        
        #paths for each dataset
        train_save_path = os.path.join(save_dir, "TrainDataset")
        valid_save_path = os.path.join(save_dir, "ValidDataset")
        test_save_path = os.path.join(save_dir, "TestDataset")

        #save in arrow format for quicker access
        train_dataset.save_to_disk(train_save_path)
        valid_dataset.save_to_disk(valid_save_path)
        test_dataset.save_to_disk(test_save_path)

        logging.info(f"Successfully saved hugging face datasets.\n")
    except Exception as e:
        logging.error(f"Failed to save hugging face datasets.")
        logging.error(f"Failed with error: {e}\n")

    

def main():
    #read in the data
    data = read_data()

    #split the data
    data = data_split(data=data)

    #encode and save the data
    encode_tensor_dataset(data, save_dir='/projects/dsci410_510/SyntheticQueryData')
    logging.info("Successfully split, encoded and embedded the data.\n")
    logging.info("Data is ready to use for Model training.")


if __name__ =="__main__":
    main()



    






