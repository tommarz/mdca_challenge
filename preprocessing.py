import gensim
from gensim.models import Word2Vec, FastText
# import tensorflow
from sklearn.feature_extraction.text import TfidfVectorizer


def create_embeddings(sentences):
    model = FastText(sentences=sentences, vector_size=128, window=5)
    model.save("fasttext.model")
    return model
