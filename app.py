from flask import Flask
from flask import request
from flask import jsonify
from gensim.models.doc2vec import Doc2Vec

app = Flask(__name__)
model = Doc2Vec.load('agh.doc2vec')


@app.route("/studies")
def studies():
    search = request.args.get('search').split(',')
    infered = model.infer_vector(search, steps=100)
    similarities = model.docvecs.most_similar([infered])
    results = [{"name": name, "similarity": similarity} for (name, similarity) in similarities]
    return jsonify(results)
