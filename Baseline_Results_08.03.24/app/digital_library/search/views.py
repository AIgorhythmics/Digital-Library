# In search/views.py

from django.shortcuts import render
from django.http import HttpResponse
# Suppose search_papers is your black-box function to fetch papers
from .utils import search_papers

def index(request):
    if request.method == "POST":
        query = request.POST.get("search")
        results = search_papers(query)  # your black-box function
        return render(request, "results.html", {"results": results})
    return render(request, "index.html")
