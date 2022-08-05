import connect
import torch
from pymilvus import Collection
from sklearn.preprocessing import normalize


# Load data corpus
data_corpus_path = '../data_corpus.pt'
data_corpus = torch.load(open(data_corpus_path, 'rb')).cpu().numpy()

# Load data corpus mapping indices
data_corpus_mapping_indices_path = '../data_corpus_mapping_indices.pt'
mapping_indices = torch.load(open(data_corpus_mapping_indices_path, 'rb'))

# Connect collection
collection_name = 'image_matching_collection'
collection = Collection(collection_name)

# Insert data to collection
msg = collection.insert([mapping_indices, data_corpus])
print(msg)