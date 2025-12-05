from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .embedder import Embedder
from .milvus_client import MilvusClient

class SearchView(APIView):
    def post(self, request, *args, **kwargs):
        # Получаем текст запроса от пользователя
        query = request.data.get("query", "")
        
        if not query:
            return Response({"error": "Запрос не может быть пустым"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Генерация embedding для запроса
        try:
            embedder = Embedder(model_name="intfloat/multilingual-e5-base")  # Указываем нужную модель
            query_embedding = embedder.encode_query(query)
        except Exception as e:
            return Response({"error": f"Ошибка при генерации embedding для запроса: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Инициализация клиента Milvus
        try:
            milvus = MilvusClient(host="standalone", port=19530)  # Указываем адрес и порт для подключения
        except Exception as e:
            return Response({"error": f"Ошибка при подключении к Milvus: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Выполнение поиска в коллекции Milvus
        collection_name = "documents"  # Убедитесь, что коллекция существует в Milvus
        try:
            # Выполняем поиск похожих векторов
            search_results = milvus.search(
                collection_name=collection_name,
                query_vectors=[query_embedding],
                top_k=3  # Количество наиболее похожих результатов
            )
            
            print(search_results)

            # Форматируем и возвращаем результаты поиска
            formatted_results = []
            for hit in search_results[0]:  # search_results - это список, полученный из Milvus
                formatted_results.append({
                    "distance": hit.get("distance", 0),  # Получаем расстояние (похожесть)
                    "text": hit.get("text", "Нет текста"),  # Получаем текст
                    "file_name": hit.get("file_name", "N/A"),  # Имя файла
                    "chunk_index": hit.get("chunk_index", -1)  # Индекс чанка
                })

            # Возвращаем отформатированные результаты
            return Response({"results": formatted_results}, status=status.HTTP_200_OK)

        
        except Exception as e:
            return Response({"error": f"Ошибка при выполнении поиска в Milvus: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
