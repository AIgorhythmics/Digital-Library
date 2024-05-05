# In search/views.py

from django.shortcuts import render
from django.http import HttpResponse
# Suppose search_papers is your black-box function to fetch papers
from .utils import search_papers, generate_summary, generate_word_cloud, preprocess_text
from django.http import JsonResponse
import re
import pandas as pd
processed_vector_location = r'E:\Github\Digital Library\Digital-Library\Baseline_Results_08.03.24\app\digital_library\search\sampled_data_20000.csv'
df = pd.read_csv(processed_vector_location)
import urllib
import re
import markdown


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
        summary = markdown.markdown(summary)
        # save it into a file
        with open('summary.txt', 'w') as f:
            f.write(summary)
        print("summary", summary)
        return JsonResponse({'summary': summary})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# views.py

import base64
from io import BytesIO

def wordcloud(request, paper_id):
    paper_id = urllib.parse.unquote(paper_id)
    print("wordcloud paper_id", paper_id)
    try:
        image_base64 = generate_word_cloud(paper_id) 
        if image_base64 is not None:
            return JsonResponse({'image': image_base64})
        else:
            return JsonResponse({'error': 'Failed to generate word cloud'}, status=500)
    except Exception as e:
        print(f"Error generating word cloud: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

from .utils import calculate_rouge_score

def get_abstract(paper_id):
    # use df which is abscrtact against the paper_id
    abstract = df[df['id'] == paper_id]['abstract'].values[0]
    print("abstract", abstract)
    return abstract

def rouge_score(request, paper_id):
    original_paper_id = paper_id.replace('_', '/')
    try:  
        abstract = get_abstract(original_paper_id) 

        with open('summary.txt', 'r') as f:
            summary = f.read()
        # preprocess the summary
        summary = preprocess_text1(summary)
        print(summary)
        rouge_scores = calculate_rouge_score(abstract, summary)
        return JsonResponse({'score': rouge_scores})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# write processing functions that remove html tags and other unwanted characters
def preprocess_text1(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text