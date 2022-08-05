import connect
from pymilvus import Collection, utility


collection_name = 'image_matching_collection'
collection = Collection(collection_name)

# Drop index if it exists
collection.drop_index()


index_params = {
    'metric_type': 'L2',
    'index_type': 'IVF_FLAT',
    'params': {
        'nlist': 1024,
    }
}

collection.create_index(
    field_name = 'embedding',
    index_params = index_params,
)