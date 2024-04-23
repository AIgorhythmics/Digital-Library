# In search/views.py

from django.shortcuts import render
from django.http import HttpResponse
# Suppose search_papers is your black-box function to fetch papers
from .utils import search_papers, generate_summary
from django.http import JsonResponse
import re
import pandas as pd
processed_vector_location = r'E:\Github\Digital Library\Digital-Library\Baseline_Results_08.03.24\app\digital_library\search\sampled_data_20000.csv'
df = pd.read_csv(processed_vector_location)



def index(request):
    if request.method == "POST":
        query = request.POST.get("search")
        results = search_papers(query, df)  
        return render(request, "results.html", {"results": results})
    return render(request, "index.html")

def paper_summary(request, paper_id):
    summary = generate_summary(paper_id)  # Function to generate summary based on paper ID
    return JsonResponse({'summary': summary})
