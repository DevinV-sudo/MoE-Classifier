<<<<<<< HEAD
# MoE-Classifier
MoE-Gating for RAG: Lightweight DistilBERT - infers routing decisions based on agentically generated training data via clustering expert knowledge embeddings to generate synthetic prompts. 
=======
# MoE-Classifier: MoE-Gating for RAG
Infers routing decisions based on synthetic user prompts that are agentically generated by clustering each RAG-Experts knowledge base.
---

### Project Overview
This project is a component of a larger system I have been devloping, which is a combination of MoE and Retreival Augmentaion where each expert is a RAG system
whose knowledge base is composed of all available class material for a specific course. This particular component is in charge of infering which of a student's courses
the student's prompt is intended to query, and selecting the appropiate expert.

### Data Generation
To generate the data required for training this classifer, I logged into the interface I built for the prior portion of this project and uploaded zip files of three courses'
material. This material includes PDF excerpts (e.g. Textbooks, Reviews, Worksheets, Midterms) and Lecture and Zoom recordings. The interface asynchronously processes all of this data
into vector embeddings stored in a pinecone index corresponding to each course.

To prepare the data, in the preprocessing directory each of these indicies are pulled down and the embedding vectors are clustered using using PCA and HDBSCAN.
The clusters are then inversed transformed back to their original embedding size, and the associated text with each clusters embedding is parsed into one text file.
This text file is then batched using langchains RecursiveTextSplitter and fed in as context to GPT-3.5-Turbo using calls to open-ai's api.

GPT returns a numbered list of n-questions for each batch for each cluster for each class, which are parsed into one dataframe with the source index name acting as the label.
Then this dataframe is split into train, val and test splits, and saved. Each y-variable is one hot encoded, and each text feature is tokenized using distil-bert's tokenizer.
Then these pairs are saved in .arrow format in the project directory.

# Data Handling




>>>>>>> e2da417 (initial commit)
