from pymilvus import Collection
import connect
import torch


# Load data corpus collection
collection_name = 'image_matching_collection'
collection = Collection(collection_name)
collection.load()


# Define search configuration parameters
search_params = {
    'metric_type': 'L2',
    'params': {
        'nprobe': 10
    }
}


# Load sample from corpus
data_corpus_path = '../data_corpus.pt'
data_corpus = torch.load(open(data_corpus_path, 'rb')).cpu().numpy()

sample = data_corpus[0]

results = collection.search(
    data = [sample],
    anns_field = 'embedding',
    param = search_params,
    limit = 10,
    expr = None,
    consistent_level = "Strong",
)

print(results)