from nltk import word_tokenize
import nltk
import os
from os import listdir
from sklearn.metrics.pairwise import linear_kernel

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import pickle


class TfidfModel:
    nltk.download('punkt')

    def __init__(self):
        self.tfidf = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english')
        self.token_dict = {}
        self.tfidf_matrix = []
        self.labels = []

    @staticmethod
    def find_similar(tfidf_matrix, input, top_n=5):
        cosine_similarities = linear_kernel(input, tfidf_matrix).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
        return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

    @staticmethod
    def tokenize(text):
        tokens = word_tokenize(text)
        stems = []
        for item in tokens:
            stems.append(PorterStemmer().stem(item))
        return stems

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def create_from_files(self, data_dir):
        self.labels = [f for f in listdir(data_dir) if f.endswith('.txt')]

        for doc in self.labels:
            with open("{}/{}".format(data_dir, doc), 'r') as file:
                text = file.read()
                (document, extension) = os.path.splitext(doc)
                self.token_dict[document] = text.lower()

        self.tfidf_matrix = self.tfidf.fit_transform(self.token_dict.values())

    def similar(self, text):
        input_vec = self.tfidf.transform([text])
        items = []
        for index, score in self.find_similar(self.tfidf_matrix, input_vec):
            items.append((self.labels[index], score))
        return items
