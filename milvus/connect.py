from pymilvus import connections, Collection

connections.connect(
    alias = 'default',
    host = 'localhost',
    port = 19530,
)

print("Connected to milvus server!!!")
