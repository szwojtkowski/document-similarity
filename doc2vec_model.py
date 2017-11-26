import os
from os import listdir
from nltk import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

data_dir = "syllabus"

labels = [f for f in listdir(data_dir)]

data = {}
for doc in labels:
    with open("{}/{}".format(data_dir, doc), 'r') as file:
        text = file.read()
        (document, extension) = os.path.splitext(doc)
        data[document] = word_tokenize(text)

tagged_documents = [TaggedDocument(words, [tag]) for (tag, words) in data.items()]

model = Doc2Vec(tagged_documents, dm=1, alpha=0.1, size=500, min_alpha=0.025, min_count=0)
# model.intersect_word2vec_format("wiki.word2vec")

model.train(tagged_documents, total_words=model.corpus_count, epochs=400)

model.save('agh.doc2vec')

