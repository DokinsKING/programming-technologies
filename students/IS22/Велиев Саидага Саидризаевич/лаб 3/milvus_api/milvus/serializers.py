from rest_framework import serializers

class CollectionSerializer(serializers.Serializer):
    collection_name = serializers.CharField(max_length=100)
    dimension = serializers.IntegerField()
    description = serializers.CharField(required=False, allow_blank=True)
    metric_type = serializers.ChoiceField(choices=["COSINE", "L2", "IP"], default="COSINE")

class SearchSerializer(serializers.Serializer):
    query_vectors = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))
    top_k = serializers.IntegerField(default=5)
    expr = serializers.CharField(required=False, allow_blank=True)
