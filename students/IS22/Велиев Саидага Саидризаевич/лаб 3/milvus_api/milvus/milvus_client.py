from pymilvus import connections, Collection, utility
from typing import List, Optional

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

    def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 5,
        expr: Optional[str] = None
    ) -> List[List[dict]]:
        """
        Поиск похожих векторов.
        
        Args:
            collection_name: Имя коллекции
            query_vectors: Векторы запросов
            top_k: Количество результатов для каждого запроса
            expr: Опциональное выражение для фильтрации (например, "text like '%python%'")
        
        Returns:
            Список результатов для каждого запроса
        """
        if not utility.has_collection(collection_name):
            raise ValueError(f"Коллекция '{collection_name}' не существует")
        
        if not query_vectors:
            raise ValueError("Список векторов запросов не может быть пустым")
        
        collection = Collection(collection_name)
        
        try:
            collection.load()
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить коллекцию '{collection_name}': {e}")
        
        # Получаем метрику из индекса коллекции
        indexes = collection.indexes
        metric_type = "COSINE"  # По умолчанию
        if indexes:
            # Берем метрику из первого индекса поля embedding
            for index in indexes:
                if index.field_name == "embedding":
                    metric_type = index.params.get("metric_type", "COSINE")
                    break
        
        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": 10}  # Количество кластеров для поиска
        }
        
        results = collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "file_name", "file_path", "chunk_index"]  # Возвращаем текст и метаданные
        )
        
        # Форматируем результаты
        formatted_results = []
        for result in results:
            hits = []
            for hit in result:  # Заменили на правильный формат
                # Теперь result - это список, и каждый hit является словарем
                hits.append({
                    "id": hit.get('id', 'N/A'),  # Получаем ID
                    "distance": hit.get('distance', 0),  # Получаем расстояние
                    "text": hit.get('text', "Нет текста"),  # Получаем текст
                    "file_name": hit.get('file_name', 'N/A'),  # Имя файла
                    "file_path": hit.get('file_path', 'N/A'),  # Путь к файлу
                    "chunk_index": hit.get('chunk_index', -1)  # Индекс чанка
                })
            formatted_results.append(hits)
        
        return formatted_results
