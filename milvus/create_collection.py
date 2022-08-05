from h11 import Data
from pydantic import Field
from pymilvus import (
    CollectionSchema, 
    FieldSchema, 
    DataType, 
    Collection,
    utility
)
import connect


collection_name = 'image_matching_collection'
# Drop collection if it exists to create a new one
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

_id = FieldSchema(
    name = '_id',
    dtype = DataType.VARCHAR,
    max_length = 255,
    is_primary = True,
)


embedding = FieldSchema(
    name = 'embedding',
    dtype = DataType.FLOAT_VECTOR,
    dim = 512
)


schema = CollectionSchema(
    fields = [_id, embedding],
    description = 'Image matching collection'
)

collection = Collection(
    name = collection_name, 
    schema = schema,
    using = 'default',
    shards_num = 10,
    consistency_level = 'Strong'
)

print(collection)






