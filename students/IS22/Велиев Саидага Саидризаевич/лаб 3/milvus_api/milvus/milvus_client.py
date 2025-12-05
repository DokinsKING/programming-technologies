from pymilvus import connections, Collection, utility

class MilvusClient:
    def __init__(self, host="standalone", port=19530, alias="default"):
        self.host = host
        self.port = port
        self.alias = alias
        self._connect()

    def _connect(self):
        """Установка подключения к Milvus."""
        try:
            connections.connect(alias=self.alias, host=self.host, port=self.port)
        except Exception as e:
            raise Exception(f"Ошибка подключения: {e}")

    def create_collection(self, collection_name, dimension, description="", metric_type="COSINE"):
        """Создание коллекции в Milvus."""
        if utility.has_collection(collection_name):
            return Collection(collection_name)
        
        fields = [
            # Определение полей коллекции
        ]
        schema = CollectionSchema(fields)
        collection = Collection(name=collection_name, schema=schema)
        # Индексы и другие параметры
        collection.create_index(field_name="embedding", index_params={"metric_type": metric_type})
        return collection

    def search(self, collection_name, query_vectors, top_k=5, expr=None):
        """Поиск в коллекции."""
        if not utility.has_collection(collection_name):
            raise ValueError(f"Коллекция {collection_name} не существует")
        
        collection = Collection(collection_name)
        results = collection.search(data=query_vectors, anns_field="embedding", param={"nprobe": 10}, limit=top_k, expr=expr)
        return results
