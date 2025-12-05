# milvus/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .milvus_client import MilvusClient
from .serializers import CollectionSerializer, SearchSerializer

milvus_client = MilvusClient()

class CreateCollectionView(APIView):
    def post(self, request):
        serializer = CollectionSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            collection = milvus_client.create_collection(
                collection_name=data['collection_name'],
                dimension=data['dimension'],
                description=data.get('description', ""),
                metric_type=data.get('metric_type', "COSINE")
            )
            return Response({"message": f"Коллекция '{collection.name}' успешно создана!"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SearchView(APIView):
    def post(self, request):
        serializer = SearchSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            results = milvus_client.search(
                collection_name=request.data.get('collection_name'),
                query_vectors=data['query_vectors'],
                top_k=data['top_k'],
                expr=data.get('expr')
            )
            return Response({"results": results}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
