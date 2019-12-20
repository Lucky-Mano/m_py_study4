import json
from pathlib import Path

import responder

from .modules import TextClassifier

api = responder.API()


@api.route("/api/v1")
def get(req, resp):
    text = req.params.get("message", "")

    clf = TextClassifier("mecab-ipadic-neologd")
    clf.load_vectorizer(Path("./vec.sav"))
    clf.load_classifier(Path("./clf.sav"))

    result = clf.predict(text)

    resp.headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Access-Control-Allow-Origin": "*",
    }

    resp.media = json.dumps({"status": "OK", "result": int(result[0])})
