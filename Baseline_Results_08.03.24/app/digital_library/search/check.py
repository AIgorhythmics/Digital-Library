id = 'math_0503092'
from wordcloud import WordCloud
import base64
import urllib
from io import BytesIO
import numpy as np
import pandas as pd
import fitz
import requests
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import os

import os
import requests
import PyPDF2

def download_and_extract_text(arxiv_id):
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Will raise an HTTPError for bad requests (400+)
        pdf_path = f"{arxiv_id}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        
        # Now extract text from the PDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except requests.exceptions.HTTPError as e:
        print(f"Failed to download or process the PDF: {e}")
        return None
    else:
        raise Exception("Failed to download PDF")


def wordcloud(paper_id):
    paper_id = urllib.parse.unquote(paper_id)
    try:
        print(paper_id)
        text = download_and_extract_text(paper_id)
        print(text)
        image = generate_word_cloud(text)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        print(({'image': image_base64}))
    except Exception as e:
        print({'error': str(e)}, status=500)
    

def preprocess_text(text):
    text = re.sub(r'##.*', '', text)
    text = re.sub(r'.*?:\s*\n', '', text)
    text = re.sub(r'\*', '', text)
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]  
    
def generate_word_cloud(text):
    text = preprocess_text(text)
    combined_text = " ".join(text)

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

wordcloud(id)