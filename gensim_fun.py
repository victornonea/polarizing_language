from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('fasttext-wiki-news-subwords-300/fasttext-wiki-news-subwords-300.gz', binary=False)
