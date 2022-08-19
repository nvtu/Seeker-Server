import __init__
from milvus import connect
from pymilvus import Collection
import json
import redis


CLIP_SERVER_URL = 'http://localhost:9001/query'
REDIS_HOST = 'localhost'
REDIS_PORT = 6379


# Milvus server setup
collection_name = 'ECIR23_Charades_v1'
collection = Collection(collection_name)
collection.load()

milvus_search_params = {
    'metric_type': 'L2',
    'params': {
        'nprobe': 10
    }
}

# Redis server setup
rd = redis.Redis(host = REDIS_HOST, port = REDIS_PORT)
if rd.ping():
    print('Redis server is running')
else:
    print('Redis server is not running')