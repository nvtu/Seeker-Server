from pkg_resources import ContextualVersionConflict
from pymilvus import Collection
import connect


collection_name = 'image_matching_collection'
collection = Collection(collection_name)

print(f"Number  of items: {collection.num_entities}")
