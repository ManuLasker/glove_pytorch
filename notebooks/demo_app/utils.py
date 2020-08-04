import subprocess
import logging
import time
import json
import mimetypes
import http.client
import warnings
import re 
import os
import random
import boto3
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from googletrans import Translator
from bs4 import BeautifulSoup
from tqdm import tqdm
from langdetect import detect
from langdetect import DetectorFactory
from nltk.translate.bleu_score import sentence_bleu


# Set detectorFactory as a non-determistic
DetectorFactory.seed = 1
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger()

NLP = spacy.load('en_core_web_sm')


def get_lang(data):
    """get language in deterministic way
    Args:
        data (String): data from blogs
    Returns:
        lang (String): lang code
    """
    lang = detect(data)
    return lang

def download_dataset(page):
    """
    Download dataset and save to a python list
    Args:
        page -- last page scarped
    Returns:
        data_temp -- python list containing dict for each blog data
    """
    sw = True
    data_temp = []
    numblog = 0
    while sw:
        try:
            conn = http.client.HTTPSConnection("koombea20stg.wpengine.com")
            conn.request("GET", f"//wp-json/wp/v2/posts?page={page}&per_page=1")
            res = conn.getresponse()
            data = res.read()
            data = json.loads(data)
            numblog += len(data)
            data_temp = data_temp + data
            page += 1
            if numblog%20 == 0:
                logger.info("Downloading blogs = {0}".format(numblog))
                time.sleep(2)
        except Exception as e:
            logger.error("Error! {0}".format(e))
            sw = False
    last_page = page - 1
    return data_temp, last_page  

def clean_html(html_content):
    """
    Clean html form of the data
    Argument:
        html_content -- Blog's content in html form
    
    Returns:
        clean_text -- python string containing the blog's
        content cleaned and parsed with the beatifulsoup html.parser method
    """
    
    clean_text = None
    soup = BeautifulSoup(html_content, "html.parser")
    clean_text = soup.get_text()
    return clean_text

def get_data_frame(page):
    """
    Clean the data and generate a pandas dataframe with the values
    Args:
        page -- last page scrapped
    Return:
        df -- pandas dataframe with all the data and sort by id
    """
    logger.info("Downloading Dataset on {0}/{1}".format("koombea.com", 
                                                        "//wp-json/wp/v2/posts?page&per_page"))
    data_temp, last_page = download_dataset(page)
    logger.info("Begin To clean datablogs and grab title and content information")

    # Clean html form of data blogs
    blogs = []
    for blog in tqdm(data_temp, desc="Cleaning html data"):
        info_blog = {}
        info_blog["id"] = blog["id"]
        info_blog["title"] = clean_html(blog["title"]["rendered"])
        info_blog["content"] = clean_html(blog["content"]["rendered"])
        info_blog["slug"] = clean_html(blog["slug"])
        blogs.append(info_blog)
    
    # Transform to a simple dataframe
    df = pd.DataFrame(blogs)
    idx_ord = df.id.sort_values(ascending=True).index
    df = df.loc[idx_ord]
    df.reset_index(drop=True, inplace=True)
    logger.info("Finish!! Donwloading the blogs")
    
    return df, last_page

def translate_spanish_data(data):
    """
    Get data if data is spanish then translate to english, otherwise do nothing
    
    Args:
        data -- string corpus
    Returns:
        translation -- translation from spanish to english
    """
    lang = detect(data)
    if lang != 'en':
        new_data = []
        translator = Translator()
        translation = translator.translate(data, dest="en", src=lang).text
    else:
        translation = data
        
    return translation

def get_regex_expression():
    """
    Generate some regex expression 
    """
    # Match non alphanumeric characters
    NON_ALPHANUMERIC_REGEX = r'[^a-zA-Z0-9À-ÿ\u00f1\u00d1\s]'
    # Match any link or url from text
    LINKS_REGEX = r'https?:\/\/.*[\r\n]'
    # Match hashtags
    HASHTAGS_REGEX = r'\#[^\s]*'
    # Match twitter accounts
    TWITTER_ACCOUNTS_REGEX = r'\@[^\s]*'
    # Match Author:
    AUTHOR_REGEX = r'author'
    # Match email 
    EMAIL_REGEX = r"\S*@\S+"
    # Group of regex
    MATCHES_GROUPED = ('({})'.format(reg) for reg in [
                                                  LINKS_REGEX, 
                                                  HASHTAGS_REGEX, 
                                                  TWITTER_ACCOUNTS_REGEX,
                                                  AUTHOR_REGEX,
                                                  EMAIL_REGEX,
                                                  NON_ALPHANUMERIC_REGEX
                                                  ])
    
    # Regex for matches group
    MATCHES_GROUPED_REGEX = r'{}'.format(('|'.join(MATCHES_GROUPED)))
    
    return MATCHES_GROUPED_REGEX

def remove_unnecesary_text(text, regex):
    """
    Remove unnecesary text using regex
    
    Args:
        text -- python string 
        regex -- python regex
    Returns:
        text -- python string
    """
    return re.sub(regex, ' ', text, flags = re.M | re.I)

# Remove all whitespace characters
def remove_whitespace(text):
    """
    Remove unnecesary whitespace
    
    Args:
        text -- python string
    Returns:
        text -- python string
    """
    return ' '.join(text.split())

def preprocess_data(text, regex, removing_stops=False, lemmatize=False):
    """
    Preprocess string data.
    Args:
        text -- A string python that is on the columns of a pandas dataframe
        regex -- Regular expression
        removing_stops -- Boolean python, if True remove english stops words
        lemmatize -- Boolean python, if True lemmatize english words
    Returns:
        text -- The Preprocess string data python
    """
    # Clean text
    text = remove_whitespace(remove_unnecesary_text(text, regex))
    
    # Tokenize the text of the blogs
    tokens = NLP(text)
    
    # Remove all punctuation marks
    tokens = [token for token in tokens if not token.is_punct]
    
    # Remove numbers or amount representation
    tokens = [token for token in tokens if not token.like_num]
    
    if removing_stops:
        # Remove stopswords
        tokens = [token for token in tokens if not token.is_stop]
        
    if lemmatize:
        # Lemmatize words
        tokens = [token.lemma_.strip().lower() for token in tokens]
    else:
        # Convert to str and lowerize
        tokens = [token.text.strip().lower() for token in tokens]
        
    tokens = [token for token in tokens if len(token)>1]
    
    return tokens

def time_counter(start_time, end_time):
    """
    Nice function to calculate the time in minutes and second of a job
    Args:
        start_time -- time before running the job
        end_time -- time after running the job
    Return:
        elapsed_mins -- time spend in minutes
        elapsed_secs -- time spend in secs
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def requests_by_slug(slug):
    """
    Make a get request by slug to staging koombea
    Args:
        slug -- string slug of the blogs
    Return:
        title_input -- blogs title
        content_input -- blogs content
        ids_input -- id blogs
    """
    logger.info("Making request by slug to koombea.com"
                "/wp-json/wp/v2/posts?slug={0}".format(slug))
    # Making connections
    conn = http.client.HTTPSConnection("koombea.com")
    conn.request("GET", "/wp-json/wp/v2/posts?slug={0}".format(slug))
    res = conn.getresponse()
    data = res.read()
    data = json.loads(data)[0]

    # Clean html
    title_input = clean_html(data["title"]["rendered"])
    content_input = clean_html(data["content"]["rendered"])
    # slug_input = str(data["slug"])
    ids_input = int(data["id"])

    return tuple((title_input, content_input, ids_input))

def download_pretrained_glove():
    """
    Download from s3 bucket to /opt/ml/model
    """
    if not os.path.exists('/opt/ml/model/glove-300.txt'):
        logger.info("Downloading pretrained model...")
        s3 = boto3.client('s3')
        s3.download_file("sagemaker-us-west-2-256305374409", 
                         "Koombea_Blogs/Blogs_Data/models/glove-pretrained/glove-300.txt",
                            "/opt/ml/model/glove-300.txt")
    else:
        logger.info("Pretrained model already downloaded...")

def show_topics(matrix_V, vocab, num_top_words=8):
    """
    Show topics using matrix factorization here you have a matrix representation of 
    docs x words using tfidf, then you can use different mathematics techniques to factorize
    de tfidf matrix in docs x words = docs x topics * topics x words, then the second matrix
    will have the topics important by word. So we can calculate the most representing words in 
    each topic
    
    Args:
        matrix_V -- matrix representation onf topics x words
        num_top_words -- int with the quantity of words
    Return:
        words -- most representing words by topics
    """
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in matrix_V])
    return [' '.join(t) for t in topic_words]


def visualize_tfidf_weights(matrix, vocab, top_n=5, total=10, width=13, height=7, div=2):
    """
    Visualiza tfidf weigths of a random sample with barplots
    
    Args:
        matrix -- matrix containing tfidf weights
        top_n -- plot the top_n words
        total -- how many we want to visualize
        width --  plot config
        height -- plot config
        div -- divisition of columns and rows on the plot, for example if total is n_samples and
               div is 2, rows: 5, col: 2
    """
    num_plot, total, div = 0, total, div
    random_sub_sample = random.choices(range(matrix.shape[0]), k=total)
    matrix_top_n_idx = matrix.argsort(axis=1)[:,-top_n:]

    total_row = total//div
    
    if total%div == 0:
        total_col = div
    else:
        total_col = div + 1
    
    plt.figure(figsize=(width, height))

    for idx_plot, idx_matrix in enumerate(random_sub_sample):
        plt.subplot(total_row, total_col, idx_plot+1)
        best_vocab = vocab[matrix_top_n_idx[idx_matrix]]
        value_vocab = matrix[idx_matrix, matrix_top_n_idx[idx_matrix]]
        sns.barplot(best_vocab, value_vocab)
        plt.ylabel("tfidf weights")

    plt.tight_layout()
    plt.show()

def visualize_tfidf_topics_weights(matrix, vocab, top_n=5, width=13, height=7, div=2):
    """
    Visualiza tfidf weigths of a topics of the matrix V, using non negative matrix factorization
    with barplots
    
    Args:
        matrix -- matrix containing tfidf weights
        top_n -- plot the top_n words
        width --  plot config
        height -- plot config
        div -- divisition of columns and rows on the plot, for example if total is n_samples and
               div is 2, rows: 5, col: 2
    """
    num_plot, total, div = 0, matrix.shape[0], div
    matrix_top_n_idx = matrix.argsort(axis=1)[:,-top_n:]

    total_row = total//div
    
    if total%div == 0:
        total_col = div
    else:
        total_col = div + 1
    
    plt.figure(figsize=(width, height))

    for idx_plot, idx_matrix in enumerate(range(matrix.shape[0])):
        plt.subplot(total_row, total_col, idx_plot+1)
        best_vocab = vocab[matrix_top_n_idx[idx_matrix]]
        value_vocab = matrix[idx_matrix, matrix_top_n_idx[idx_matrix]]
        sns.barplot(best_vocab, value_vocab)
        plt.ylabel("tfidf weights")

    plt.tight_layout()
    plt.show()

def plots_blogs_embedding(tfidf_matrix, tsne_matrix, vocab, clusters, 
                          total_samples = 300, width=8, height=5):
    """
    Plot cluster of tsne values, with annotation of the heights weights on
    tfidf matrix
    
    Args:
        tfidf_matrix -- tfidf-weights
        tsne_matrix -- tsne matrix onto two dimensional space
        vocab -- array of vocab used by tfidf
        clusters -- clusters with kmeans
        total_samples -- total samples to show on the plot
        width -- width of the plot
        height -- height of the plot
    """
    plt.figure(figsize = (width, height))
    matrix_top_n = tfidf_matrix.toarray().argsort(axis=1)[:,-2:]
    total = total_samples
    scatter = plt.scatter(tsne_matrix[:total,0], tsne_matrix[:total,1], 
                         c = clusters[:total], s=50, cmap='viridis')
    words = [list(vocab[i]) for i in matrix_top_n][:total]
    for i, word in enumerate(words):
        plt.annotate(" ".join(word), xy=(tsne_matrix[i, 0], tsne_matrix[i, 1]))
    plt.title("blogs using 2 tsne components with annotations")
    plt.show()

def visualize_words_embedding(tsne_matrix, vocab, width=12, height=12, 
                              total=100, title="words using 2 tsne comp"):
    """
    visualize word embedding of a w2v model using tsne manifold
    
    Args:
        tsne_matrix -- tsne maniforld with 2 components
        vocab -- array with vocab
        width -- width plot
        height -- height plot
        total -- total vocab
        title -- title plot
    """
    plt.figure(figsize=(width, height))
    random_pick =  random.choices(range(tsne_matrix.shape[0]), k=total)
    scatter = plt.scatter(tsne_matrix[random_pick,0], tsne_matrix[random_pick, 1],
                         cmap="viridis")
    words = vocab[random_pick]
    for r_idx, word in zip(random_pick, words):
        plt.annotate(word, xy=(tsne_matrix[r_idx,0], tsne_matrix[r_idx, 1]))
    plt.title(title)
    plt.show()

def compute_similarities(task_length, fse_mosel):
    """
    Compute similarities using cosine scores
    
    Args:
        fse_model -- fast sentence model
        task_length -- size of our data
    Return:
        similarity_score -- list containing the similarity score
    """
    sims = []
    for i in tqdm(range(task_length)):
        for j in range(task_length):
            sims.append(np.abs(fse_mosel.sv.similarity(i, j)))
    return sims