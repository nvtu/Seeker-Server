import connect
import torch
from pymilvus import Collection
import json
from tqdm import tqdm
import os
from sklearn.preprocessing import normalize


# Connect collection
collection_name = 'ECIR23_Charades_v1'
collection = Collection(collection_name)

data_corpus_path = '/mnt/DATA/nvtu/ECIR23/video_based_embedding_features'
index_path = os.path.join(os.path.dirname(data_corpus_path), 'Charades_v1_metadata.json')

# Load mapping indices
mapping_indices = json.load(open(index_path))

# Load data corpus
for video_name in tqdm(os.listdir(data_corpus_path)):
    video_path = os.path.join(data_corpus_path, video_name)
    video_name = os.path.splitext(video_name)[0] # Remove extension
    data = torch.load(open(video_path, 'rb')).cpu().numpy()
    mapping = mapping_indices[video_name]

    # Insert data to collection
    msg = collection.insert([mapping, data])
    # print(msg)