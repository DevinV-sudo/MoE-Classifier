#imports
from dotenv import load_dotenv
import os
from pinecone import Pinecone
import numpy as np
import logging
import re
import json
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import openai
import pandas as pd

#set up logging config
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

#load env vars
load_dotenv()

#create pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

#create openai client
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def load_clusters():
    #create pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    #pull down index names
    available_indices = pc.list_indexes()
    index_names = [item.name for item in available_indices]
    directory_names = [name.replace("-", "_").replace(" ", "_")+"_clusters" for name in index_names]

    #base path to clusters
    base_path = "data"

    cluster_data = []
    for name in directory_names:
        #generate paths to the clustering json files
        dir_path = os.path.join(base_path, name)
        file_path = os.path.join(dir_path, "clusters.json")

        #check if the file exists
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    clusters = json.load(f)
                    cluster_data.append(clusters)
                    logging.info(f"Successfully loaded {file_path}\n")

            except Exception as e:
                logging.error("Failed to open clusters: {e}\n")
                continue
        else:
            logging.warning(f"No cluster exists at path: {file_path}\n")

    return cluster_data

def extract_text(cluster):
    #one index at a time
    index_name = cluster[0].get("index_name")

    #iterate through cluster and group by cluster label
    cluster_dict = {}
    for vector in cluster:
        cluster_label = vector.get("cluster_label")

        # if the label has not been seen intialize as empty string
        if cluster_label not in cluster_dict:
            cluster_dict[cluster_label] = ""
        
        #extract text from vector
        vector_text = vector["text"]

        #add the text padded by a new line to the dictionary
        cluster_dict[cluster_label] += f"\n {vector_text}"

    return index_name, cluster_dict

    
def load_tokenizer(tokenizer_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logging.info(f"Tokenizer {tokenizer_name} loaded successfully\n")
        return tokenizer
    
    except Exception as e:
        logging.error(f"Tokenizer: {tokenizer_name} could not be loaded: {e}\n")
        return None

def check_cxt_length(context, tokenizer, max_content_length):
    try:
        tokens = tokenizer.encode(context, add_special_tokens=True)
        token_count = len(tokens)

        if token_count <= max_content_length:
            logging.info(f"Context is within limit: token count = {token_count}, max tokens = {max_content_length}\n")
            return True
        
        else:
            logging.info(f"Token count exceeded: token count = {token_count}, max tokens = {max_content_length}\n")
            return False
        
    except Exception as e:
        logging.error(f"Error checking context length: {e}\n")
        return False

def batch_context(context, max_content_length):
    # Create Document objects from the context string
    docs = [Document(page_content=context)]
    
    # initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_content_length//3,
        chunk_overlap=100,
        strip_whitespace=True
    )

    # apply to context
    split_docs = text_splitter.split_documents(docs)
    text = [doc.page_content for doc in split_docs]
    return text

def generate_questions(index_name, cluster_dict, n_questions, max_content_length, model_name):
    #dictionary to store outputs
    data = {}
    data["Index"] = index_name

    for cluster_label in cluster_dict.keys():
        logging.info(f"Generating questions for cluster: {cluster_label}\n")

        #pull out the text
        text = cluster_dict[cluster_label]

        #just batch the text
        docs = batch_context(text, max_content_length)

        #Determine the number of questions to generate
        question_count = n_questions // len(docs)

        #storage for cluster output
        output_questions = []

        #iterate through docs (if multiple)
        for doc in docs:
            context = doc

            #chat-gpt prompt
            prompt = f"""
                    You are generating student questions based on the following class material.  
                    Your task is to generate {question_count} thoughtful and diverse questions a student might ask.  

                    ### Guidelines:  
                    - Output only **a numbered list of questions**.  
                    - Make sure questions cover different aspects of the material.
                    - If the context is unclear or too brief, still attempt to generate relevant questions.  

                    ### Example Output:  
                    1. What is gravity?
                    2. How does photosynthesis work?
                    3. Why is the sky blue?
                    
                    ### Class material:
                    {context}
                    """
            
            try:
                #making api call
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert educator."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.7)
                

            except Exception as e:
                logging.error(f"Error making OpenAI API call: {e}\n")
                return []
            
            #extract just the text output
            response_text = response.choices[0].message.content

            #convert the numbered list to a python list
            questions = re.findall(r'\d+\.\s*(.*)', response_text)
            
            #add it to the output
            output_questions.extend(questions)

        #attach this clusters questions to the dictionary
        data[f"Cluster_{cluster_label}"] = output_questions
    
    #return the final dictionary
    return data
        
def driver_function(model_name, max_content_length, tokenizer_name):
    ''' This function drives the data generation process'''
    #load the clusters
    clusters = load_clusters()

    #dictionary for dataframes
    data_frames = {}

    for cluster in clusters:
        try:
            logging.info(f"Preparing cluster....\n")

            #extract text
            index_name, cluster_data = extract_text(cluster)
            
            

            #generate questions
            cluster_questions = generate_questions(index_name=index_name, cluster_dict=cluster_data,
                                                    n_questions=1000,
                                                    max_content_length=max_content_length,
                                                    model_name=model_name)
            
            #store in a dataframe
            logging.info(f"Generating Data Frames.....\n")
            questions = []   

            #extract keys
            question_keys = [key for key in cluster_questions.keys() if key != "Index"]

            #iterate through keys
            for key in question_keys:
                for question in cluster_questions[key]:
                    questions.append({"Question":question, "Label": index_name})
            
            #generate a dataframe with the questions and index label
            df = pd.DataFrame(questions)

            #store in dictionary
            data_frames[cluster_questions["Index"]] = df

        except Exception as e:
            logging.error(f"failed to generate questions for index: {index_name}, Failed with: {e}\n")
            continue

    #concatenate the stored dataframes
    logging.info(f"Mergin DataFrames")
    frames = [value for value in data_frames.values()]
    data_result = pd.concat(frames, ignore_index=True)

    #return the result
    logging.info(f"Successfully merged dataframes")
    return data_result

def save_frames(data_frame):
    ''' This function simply saves the out put of the driver'''
    base_path = "data"

    #check if the data path exists
    if os.path.isdir(base_path):
        logging.info(f"Base path exists: Storing Frames")

        frame_path = "question_data"
        save_path = os.path.join(base_path, frame_path)
        os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, "QuestionData.csv")

        try:
            data_frame.to_csv(file_path, index=False)
            logging.info("File saved successfully.\n")
        
        except Exception as e:
            logging.error(f"Failed to save file: {e}\n")

    else:
        logging.error(f"Base path does not exist.\n")

def main():
    #model name
    model_name = "gpt-3.5-turbo"
    logging.info(f"Using model: {model_name}\n")

    #for llama 3.2 (according to openai)
    max_content_length = 4096
    logging.info(f"Max content length: {max_content_length}\n")

    #tokenizer name (according to openai, the tokenizer is the same as gp2)
    tokenizer_name = "gpt2"
    logging.info(f"Tokenizing with: {tokenizer_name}\n")

    #generate questions
    results = driver_function(model_name=model_name, max_content_length=max_content_length, tokenizer_name=tokenizer_name)

    #save the dataframes
    save_frames(results)


if __name__ == "__main__":
    main()
















