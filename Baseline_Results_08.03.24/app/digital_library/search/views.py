# In search/views.py

from django.shortcuts import render
from django.http import HttpResponse
# Suppose search_papers is your black-box function to fetch papers
from .utils import search_papers, generate_summary, generate_word_cloud
from django.http import JsonResponse
import re
import pandas as pd
processed_vector_location = r'E:\Github\Digital Library\Digital-Library\Baseline_Results_08.03.24\app\digital_library\search\sampled_data_20000.csv'
df = pd.read_csv(processed_vector_location)
import urllib


def index(request):
    if request.method == "POST":
        query = request.POST.get("search")
        results = search_papers(query, df)  
        return render(request, "results.html", {"results": results})
    return render(request, "index.html")

def paper_summary(request, paper_id):
    original_paper_id = paper_id.replace('_', '/')
    print("original", original_paper_id) # Function to generate summary based on paper ID
    try:  # Add error handling
        summary = generate_summary(original_paper_id) 
        return JsonResponse({'summary': summary})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# views.py

import base64
from io import BytesIO

def wordcloud(request, paper_id):
    paper_id = urllib.parse.unquote(paper_id)
    try:
        image = generate_word_cloud(paper_id)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return JsonResponse({'image': image_base64})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
# views.py

from .utils import calculate_rouge_score

def rouge_score(request, paper_id):
    try:
        score = calculate_rouge_score(paper_id)
        return JsonResponse({'score': score})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

