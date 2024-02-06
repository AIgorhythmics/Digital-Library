# Digital-Library
Application to search scientific papers and graphically summarize them using LLM using Information Retrieval concepts

## What is the problem identified in this project?

In this problem, we aim to build a search engine where the user inputs their query, and the search engine retrieves relevant research papers. After retrieving these results, we use an LLM and other AI techniques to summarise the paper. Apart from this, we also aim to present the summary in a graphical manner that makes it easier for the user to decide whether the paper is relevant to what he was looking for or not.

### The problem identified here is:

Reading a research paper requires **a good amount of time**. Sometimes, a person reads the abstract and decided to read the paper. Later on, they find out that the **paper was irrelevant and lose a lot of time**. 

Digital Libraries query the keyword and then provide results. These results only contain the title and the abstract of the paper. These abstracts may or may not contain results and techniques used in the paper, making it difficult for the user to assess the relevance of the paper.

We aim to **summarize the content of the papers retrieved**, including the techniques used and the results obtained. Apart from this, we also aim to **generate a graphical representation** of the summary (word clouds etc.) to** make it easier for the user to decide if the paper is worth their time**.

## Is there any related work?

There were some related works to research paper information retrieval systems but they are mostly focussed on library and digital media.

- “A Full Text Retrieval System in a Digital Library Environment" focuses on the design and implementation of a Full Text Retrieval System (FTRS) within the context of a digital library environment. The primary objective is to address the challenges associated with information retrieval (IR) in digital libraries, such as inefficient search processes and inadequate results.
It uses Boolean model, vector space model (VSM), and language models which mainly implements features such as indexing, database design, user interface design, and system implementation. [A Full Text Retrieval System in a Digital Library Environment](https://www.scirp.org/journal/paperinformation?paperid=62727)

- “Multimedia Digital Information Retrieval System” focuses on retrieval of digital data , analyzing of data reduction and document storage. It also focuses on MMDBMS architecture where large sets of digital media are taken into account and used accordingly. Content-based information retrieval research focus on the mechanism that allows the user to effectively find reusable multimedia objects, including pictures, sound, video, and other forms. After the successful retrieval, the database interface should help the user to compose/decompose multimedia documents. [Multimedia Digital Information Retrieval System](https://www.researchgate.net/publication/295419193_Digital_Information_Retrieval)

- “Rule Based Metadata Extraction Framework from Academic Articles” focuses on the efficient extraction and management of metadata from scientific articles encompassing elements like titles, abstracts, keywords, body texts, conclusions, and references. This process is essential for facilitating access to critical scientific knowledge for researchers and librarians alike. Despite the vital role of academic social networks and digital library systems in providing services for the extraction, organization, and retrieval of academic data, these services often come with limitations, including cost, open-source availability, performance issues, and restrictions on the quantity of PDF files that can be processed. [Rule Based Metadata Extraction Framework from Academic Articles](https://arxiv.org/abs/1807.09009)


## How different is your idea from theirs?
The novelty in our proposal is that we aim to offer a summary of top ranked search results based on an LLM. This offers the user further ease of use when searching as users will be further able to filter top ranked results based on their summaries to understand better and find relevant information. This approach is fast being adopted in search engines like Bing and Google where Artificial intelligence may extract relevant information from your query and match it to web pages and their content. But big Databases like PubMed and ArXiv don't offer such functionalities. We also aim to offer analysis and holistic graphical representations of the subject matter and paper data so that one can view important information regarding the paper at a glance. For example, if you are searching for ML-based research in a new field, alongside the methods used, the metrics such as accuracy and graphs will be of relevance too, they too should be provided. This will create an engaging interaction with the search engine, as one problem with reading academic papers is that it's time-consuming, and one may struggle to find exactly what they are looking for.

## Why is this problem important ???
Various platforms provide users with relevant research papers based on their input. When it comes to understanding the paper thoroughly, some users might find it difficult, even with the abstract provided by the paper. We aim to bridge this gap and provide a better heuristic summary along with a graphical representation of the paper's statistics.

## How will you evaluate your work?
The evaluation can be divided into quite a few parameters mentioned below:
- **1. Accuracy of the trained language model:**  Whether the provided results after a query search are relevant to the users or not. If the results are satisfactory, whether the summary provided by the model is sufficiently detailed or not.
- **2. User Feedback:** A user can choose to rate each search query according to the results provided by our application.
- **3. Response Time and System Scalability:** Whether the designed application is responsive enough to provide the results of a search query in a reasonable time. Furthermore, How will the application deal with new data? We can assess the system’s ability to handle an increasing number of queries, papers, and users without significant degradation in performance.  
1 and 3 are quantitative metrics while 2 is qualitative, which might also include usability testing in the later part of our project.


## What techniques/algorithms will you use/develop to solve the problem?
- Data Collection: 
  - Web Scraping:  We are going to collect data from various academic databases, journal websites.
    Dataset Link : https://www.tensorflow.org/datasets/catalog/scientific_papers
  - APIs Utilization: Utilize APIs provided by academic databases like PubMed, Google Scholar, arXiv, etc., for accessing research papers programmatically. 
- Ranking: 
  - Vector Space Model: We will represent documents and queries as vectors in a high-dimensional space and compute cosine similarity between them.
  - BM25: A probabilistic retrieval model that takes into account term frequency and document length normalization.
  - Learning to Rank: Train a machine learning model to rank search results based on relevance features extracted from documents and queries.
- Summarization: 
  - Extractive Summarization: Select important sentences or phrases from the document to form a summary. You can use techniques like TextRank or graph-based algorithms for this purpose. 
  - Abstractive Summarization: Generate a summary by paraphrasing and rephrasing the content of the document using neural network-based models such as LLMs (Large Language Models) like GPT (Generative Pre-trained Transformer) or BERT (Bidirectional Encoder Representations from Transformers) or LLAMA.
- Frontend Development: 
  - Web Development Frameworks: Using frameworks like Flask (Python), Express (Node.js), or Django (Python) for building the frontend of the search engine. 
  - User Interface Design: A user-friendly interface to input search queries and display search results and summaries effectively.

