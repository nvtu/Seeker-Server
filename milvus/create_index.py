import connect
from pymilvus import Collection, utility


collection_name = 'ECIR23_Charades_v1'
collection = Collection(collection_name)

# Drop index if it exists
# collection.drop_index()


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