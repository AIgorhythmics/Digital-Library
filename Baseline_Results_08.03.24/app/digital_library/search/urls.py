from django.urls import path
from .views import paper_summary, wordcloud
from . import views
from .utils import search_papers, generate_summary, preprocess_and_clean, tokenize
from django.urls import re_path

urlpatterns = [
    path('', views.index, name='index'),
    path('summary/<str:paper_id>', paper_summary, name='paper_summary'),
    path('rouge-score/<str:paper_id>/', views.rouge_score, name='rouge_score'),
    path('wordcloud/<str:paper_id>/', wordcloud, name='wordcloud'), 
]

