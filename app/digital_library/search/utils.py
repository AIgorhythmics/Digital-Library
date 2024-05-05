


    # top_results = df.sort_values(by='similarity', ascending=False).head(5)
    # return top_results[['id', 'title', 'abstract']].to_dict(orient='records')




from adapters import AutoAdapterModel
from transformers import AutoTokenizer
import arxiv
import json
import torch
import pandas as pd
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from io import BytesIO
import base64
# from django.conf import settings
# file_path = os.path.join(settings.BASE_DIR, 'search')



model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
token_specter=AutoTokenizer.from_pretrained('allenai/specter2_base')
adapter_name = model.load_adapter("allenai/specter2", source="hf", set_active=True)
model.eval()

def preprocess_and_clean(text):
    text = text.replace('\n', " ").strip().lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()


def tokenize(text):
    # Use NLTK's tokenizer for tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming using Porter Stemmer
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

#load the model and tokenizer


lambd=0.8

def get_vectorize(query, model, vectorizer):
    processed_query = preprocess_and_clean(query)
    bow = vectorizer.transform([processed_query])
    _q = query + token_specter.sep_token
    inp = token_specter([_q], padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        output = model(**inp)
        embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings, bow

def cosine_sparse(v1, v2):
    # Ensure v1 and v2 are in correct format, v1 and v2 should be 2D arrays
    return cosine_similarity(v1, v2)[0][0]

def cosine_dense(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-7)

def get_scores(dictionary, query, vectorizer):
    Dense_Q, Sparse_Q = get_vectorize(query, model, vectorizer)
    Sparse_Q = normalize(Sparse_Q, norm='l2', axis=1)  # Normalize sparse query
    Dense_Q = Dense_Q.flatten()
    Dense_Q = (Dense_Q - Dense_Q.mean()) / Dense_Q.std()  # Normalize dense query

    new_dic = {}
    for k, v in dictionary.items():
        d = np.array(v['Dense']).flatten()
        d = (d - d.mean()) / d.std()  # Normalize dense vector from dictionary
        s = normalize(v['Sparse'], norm="l2", axis=1) if not isinstance(v['Sparse'], csr_matrix) else v['Sparse']
        score = lambd * cosine_similarity(Dense_Q.reshape(1, -1), d.reshape(1, -1))[0][0] + (1 - lambd) * cosine_sparse(Sparse_Q, s.reshape(1, -1))
        new_dic[k] = score
    top_5 = sorted(new_dic.items(), key=lambda x: x[1], reverse=True)[:5]
    print(top_5)
    return dict(top_5)



# locations
# processed_vector_location = r'E:\Github\Digital Library\Digital-Library\Baseline_Results_08.03.24\app\digital_library\search\processed_df_20000.pkl'
processed_vector_location = r'E:\Github\Digital Library\Digital-Library\Baseline_Results_08.03.24\app\digital_library\search\sampled_data_20000.csv'

vectorizer_location = r'E:\Github\Digital Library\Digital-Library\Baseline_Results_08.03.24\app\digital_library\search\vectorizer.pkl'
embedding_dict_location =r'E:\Github\Digital Library\Digital-Library\Baseline_Results_08.03.24\app\digital_library\search\embedding_dict_20000.pkl'

#load csv to dataframe
df = pd.read_csv(processed_vector_location)

with open(vectorizer_location, 'rb') as f:
    vectorizer = pickle.load(f)

with open(embedding_dict_location, 'rb') as f:
    embedding_dict = pickle.load(f)

def search_papers(query,df):
    scores = get_scores(embedding_dict, query, vectorizer)
    df['id'] = df['id'].astype(str)
    df['similarity'] = df['id'].map(scores)
    df = df.dropna(subset=['similarity'])
    top_results = df.sort_values(by='similarity', ascending=False).head(5)
    return top_results[['id', 'title', 'abstract','similarity']].to_dict(orient='records')


import requests
import fitz  # PyMuPDF

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
import vertexai

from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

def generate_summary(arxiv_id):
    # Download the PDF and extract its text
    paper_text = download_and_extract_text(arxiv_id)
    if not paper_text:
        return "No summary available due to download or text extraction failure."

    # Initialize Vertex AI and prepare for summary generation
    vertexai.init(project="codev-418513", location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-preview-0409")
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }


    text_prompt = f"provide the summary of the following paper keep it brief and include introduction and results\n\n{paper_text}"
    responses = model.generate_content(
        [text_prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    # Concatenate the responses and return the summary
    summary_text = "".join(response.text for response in responses)
    return summary_text

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
    print('top_features',top_features)
    # Generate word cloud
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_features))
        print(type(wordcloud))  # Should be <class 'wordcloud.wordcloud.WordCloud'>
    
        wordcloud_image = wordcloud.to_image()
        print(type(wordcloud_image))  # Should be <class 'PIL.Image.Image'>
        image_io = BytesIO()
        wordcloud.to_image().save(image_io, 'PNG')
        image_io.seek(0)

        image_base64 = base64.b64encode(image_io.getvalue()).decode('utf-8')
        
        return image_base64
    except Exception as e:
        print(f"Failed to generate word cloud: {e}")
        return None
def preprocess_text(text):
    text = re.sub(r'##.*', '', text)
    text = re.sub(r'.*?:\s*\n', '', text)
    text = re.sub(r'\*', '', text)
    lines = text.split('\n')
    return [line.strip() for line in lines if line.strip()]

from rouge import Rouge

def calculate_rouge_score(abstract, summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, abstract)
    return scores
