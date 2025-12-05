from django.urls import path
from .views import CreateCollectionView, SearchView

urlpatterns = [
    path('create_collection/', CreateCollectionView.as_view(), name='create_collection'),
    path('search/', SearchView.as_view(), name='search'),
]
