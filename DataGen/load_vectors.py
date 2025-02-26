from dotenv import load_dotenv
import os
from pinecone import Pinecone, Index
import numpy as np
import hdbscan
import logging
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import json

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

available_indices = pc.list_indexes()
index_names = [item.name for item in available_indices]



#pull index names
def get_vector_ids(index_name):
    '''Given a Index Name fetch all vector ids'''
    #generate pinecone index
    try:
        logging.info(f"Extracting vector ID's from {index_name}\n")
        index = pc.Index(index_name)

        #collect vector ids
        vec_ids = []
        for ids in index.list():
            vec_ids.extend(ids)
        logging.info(f"Successfully retreived vector ID's")

        #return retreived ID's
        return vec_ids
    
    except Exception as e:
        logging.error(f"Error retreiving vector ID's: {e}")
        return None
    
    

#fetch_vectors
def fetch_vectors(index_name, vector_ids, batchsize=100):
    '''Given vector ids, and index name fetch the vector embeddings'''
    try:
        logging.info(f"Extracting vectors by ID\n")
        index = pc.Index(index_name)

        all_vectors = []
        all_vectors_text=[]

        for i in range(0, len(vector_ids), batchsize):
            batch = vector_ids[i:i+batchsize]
            logging.info(f"Fetching batch...\n")

            fetch_response = index.fetch(ids=batch)
            vectors_responses = fetch_response.vectors

            if not vectors_responses:
                logging.warning(f"No vectors found in batch {i//batchsize + 1}\n")
                continue

            all_vectors.extend([vectors_responses[vec_id]["values"] for vec_id in vectors_responses])
            all_vectors_text.extend([vectors_responses[vec_id]["metadata"]["text"] for vec_id in vectors_responses])

        vector_data = np.array(all_vectors)
        text_data = np.array(all_vectors_text)
        logging.info(f"Successfully extracted {vector_data.shape[0]} vectors\n")

        return vector_data, text_data

    except Exception as e:
        logging.error(f"Error extracting vectors from ID's: {e}\n")
        return None


def select_pca_components(data, var_thresh=0.95):
    '''
    Determine the optimal number of components for PCA,
    i.e the number of components that captures around 95%
    of the variance within the text embedding space
    '''
    try:
        logging.info(f"Calculating optimal number of components...\n")
        #fit PCA to the extracted vectors
        pca = PCA()
        pca.fit(data)

        #calculate the cumulative variance and select components
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(cumulative_variance >= var_thresh) + 1

        #return components and variance
        logging.info(f"Optimal number of components: {num_components}\n")
        return num_components
    
    except Exception as e:
        logging.error(f"Error calculating number of components: {e}\n")
        return None

def reduce_dim_clustering(n_components,  embeddings):
    '''
    reduce the vector space dimensions to speed
    up the clustering process.
    '''
    #initialize PCA
    pca = PCA(n_components=n_components)

    try:
        logging.info(f"Transforming Embeddings")
        X_pca = pca.fit_transform(embeddings)

        #cosine similarity matrix of reduced vectors
        logging.info(f"Calculating cosine distance matrix\n")
        cosine_dist_mat = cosine_distances(X_pca)

        #Cluster reduced vectors
        try:
            logging.info(f"Clustering...\n")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric="precomputed")
            labels = clusterer.fit_predict(cosine_dist_mat)

            clusters = {}
            cluster_indices = {}

            for i, label in enumerate(labels):
                if label==-1:
                    continue
                if label not in clusters:
                    clusters[label] = []
                    cluster_indices[label] = []
            
                clusters[label].append(X_pca[i])
                cluster_indices[label].append(i)

            clusters = {k: np.array(v) for k, v in clusters.items()}
            return clusters, cluster_indices, pca
        
        except Exception as e:
            logging.error(f"Error clustering reduced vectors: {e}\n")
            return None, None, None
    except Exception as e:
        logging.error(f"Error transforming vector\n")
        return None, None, None

def reverse_transform(clusters, pca):
    ''' Apply the inverse PCA transformation
    such that the vectors that have been clustered
    are in the original embedding dimension.'''

    try:
        logging.info(f"Reversing PCA Transformation...\n")
        reconstructed_clusters = {label: pca.inverse_transform(embeddings)
                                  for label, embeddings in clusters.items()}
        logging.info(f"Successfully reversed PCA transformation.\n")
        return reconstructed_clusters
    except Exception as e:
        logging.error(f"Error Reversing PCA transformation: {e}\n")
        return None

def save_clusters(clusters, index_name):
    ''' This function just saves the clustered embeddings'''
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    #replace any spaces or '-' with '_'
    new_index_name = index_name.replace('-', '_').replace(' ', '_')
    cluster_dir = os.path.join(data_dir, f"{new_index_name}_clusters")
    os.makedirs(cluster_dir, exist_ok=True)

    #file save path
    save_path = os.path.join(cluster_dir, "clusters.json")

    #attempt to save the clusters
    try:
        logging.info(f"Attempting to save {new_index_name} clusters......\n")
        with open(save_path, 'w') as f:
            json.dump(clusters, f, indent=1)
        logging.info(f"Successfully saved Clusters.\n")

    except Exception as e:
        logging.error(f"Error saving clusters: {e}\n")

def main():
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

    available_indices = pc.list_indexes()
    index_names = [item.name for item in available_indices]

    for index_name in index_names:
        logging.info(f"Processing Index: {index_name}...\n")
        #fetch the vector ids from index
        vector_ids = get_vector_ids(index_name)

        #fetch the vectors corresponding to ids
        vector_data, text_data = fetch_vectors(index_name, vector_ids)

        #calculate num components that represent 95% of data variance
        n_components = select_pca_components(data=vector_data, var_thresh=0.95)

        #reduce dimensionality and cluster
        clusters, cluster_indices, pca = reduce_dim_clustering(n_components, vector_data)

        if clusters is None:
            logging.error(f"Found no clusters in {index_name}, skipping index\n")
            continue

        #reverse transformation back to embedding dim
        reconstructed_clusters = reverse_transform(clusters, pca)

        #add the index name to each cluster
        new_index_name = index_name.replace("-", "_").replace(" ", "_")
        reconstructed_clusters_list = []

        for label, embeddings in reconstructed_clusters.items():
            for idx, embedding in zip(cluster_indices[label], embeddings):
                reconstructed_clusters_list.append({
                    "cluster_label": int(label),
                    "embedding": embedding.tolist(),
                    "index_name": str(new_index_name),
                    "text": text_data[idx]
                })

        #save clusters
        save_clusters(reconstructed_clusters_list, new_index_name)

if __name__ == "__main__":
    main()
    
