from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
from gensim.models.doc2vec import Doc2Vec
from tfidf_model import TfidfModel

app = Flask(__name__)
cors = CORS(app)
doc2vec = Doc2Vec.load('agh.doc2vec')
tfidf = TfidfModel.load('agh.tfidf')


def similarities2obj(similarities):
    results = []
    for (info, similarity) in similarities:
        (departament, degree, name) = info.split("|")
        result = {
            "name": name,
            "departament": departament,
            "degree": degree,
            "similarity": similarity
        }
        results.append(result)
    return results


@app.route("/studies")
def studies():
    search = request.args.get('search').split(',')
    text = ' '.join(search)
    similarities = tfidf.similar(text, topn=50)
    return jsonify(similarities2obj(similarities))


@app.route("/studies2")
def studies2():
    search = request.args.get('search').split(',')
    infered = doc2vec.infer_vector(search, steps=100)
    similarities = doc2vec.docvecs.most_similar([infered], topn=50)
    return jsonify(similarities2obj(similarities))
