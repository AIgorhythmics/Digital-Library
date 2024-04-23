from django.urls import path
from .views import paper_summary
from . import views
from .utils import search_papers, generate_summary, preprocess_and_clean, tokenize
urlpatterns = [
    path('', views.index, name='index'),
    path('summary/<str:paper_id>/', paper_summary, name='paper_summary')
]

