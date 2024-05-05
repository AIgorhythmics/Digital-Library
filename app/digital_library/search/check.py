
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
import arxiv
import json
import torch
import pandas as pd
import requests
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import fitz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from io import BytesIO
import base64

def preprocess_text(text):
    text = re.sub(r'##.*', '', text)
    text = re.sub(r'.*?:\s*\n', '', text)
    text = re.sub(r'\*', '', text)
    lines = text.split('\n')
    return [line.strip() for line in lines if line.strip()]

def generate_word_cloud(paper_id):
    paper_text = download_and_extract_text(paper_id)
    if paper_text is None:
        return None
    
    paper_text = preprocess_text(paper_text)
    combined_text = " ".join(paper_text)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([combined_text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

    sorted_indices = np.argsort(tfidf_scores)[::-1]
    top_features = feature_names[sorted_indices][:20]

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_features))

    # Save to BytesIO object
    image_io = BytesIO()
    wordcloud.to_image().save(image_io, 'PNG')
    image_io.seek(0)  # Go to the beginning of the BytesIO stream

    # Base64 encode
    image_base64 = base64.b64encode(image_io.getvalue()).decode('utf-8')

    return image_base64

import os

def download_and_extract_text(arxiv_id):
    pdf_path = f"{arxiv_id}.pdf"
    
    # Check if file already exists
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
    else:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()  # Will raise an HTTPError for bad requests (400+)
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            pdf_content = response.content
        except requests.exceptions.HTTPError as e:
            print(f"Failed to download or process the PDF: {e}")
            return None
    
    # Now extract text from the PDF content
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()  # Close the document
    
    return text

    
paper_id = "1707.09562"
image = generate_word_cloud(paper_id)
image_data = base64.b64decode(image)
# save the image to a file
with open("E:\\wordcloud.png", "wb") as f:
    f.write(image_data)