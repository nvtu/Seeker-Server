from pkg_resources import ContextualVersionConflict
from pymilvus import Collection
import connect


collection_name = 'ECIR23_Charades_v1'
collection = Collection(collection_name)

print(f"Number  of items: {collection.num_entities}")
