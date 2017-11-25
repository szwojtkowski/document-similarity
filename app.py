from flask import Flask
from flask import request
from flask import jsonify
from gensim.models.doc2vec import Doc2Vec
from tfidf_model import TfidfModel

app = Flask(__name__)
doc2vec = Doc2Vec.load('agh.doc2vec')
tfidf = TfidfModel.load('agh.tfidf')


@app.route("/studies")
def studies():
    search = request.args.get('search').split(',')
    text = ' '.join(search)
    similarities = tfidf.similar(text)
    results = [{"name": name, "similarity": similarity} for (name, similarity) in similarities]
    return jsonify(results)


@app.route("/studies2")
def studies2():
    search = request.args.get('search').split(',')
    infered = doc2vec.infer_vector(search, steps=100)
    similarities = doc2vec.docvecs.most_similar([infered])
    results = [{"name": name, "similarity": similarity} for (name, similarity) in similarities]
    return jsonify(results)
