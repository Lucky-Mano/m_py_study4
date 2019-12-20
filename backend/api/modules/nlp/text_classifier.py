import pickle
from pathlib import Path
from typing import Iterable, List

from mecab2pandas import MecabParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


class TextClassifier:
    def __init__(self, dic_name) -> None:
        self.parser = MecabParser(dic_name)
        self.vectorizer: TfidfVectorizer = None
        self.clf: SVC = None

    def wakachi(self, data: List[str]) -> Iterable:
        parsed_list: List[List[str]] = []
        for text in data:
            parsed = self.parser.parse(text)

            row_words: List[str] = []
            for _, row in parsed.iterrows():
                row_words.append(
                    row["original_form"]
                    if row["original_form"] is not None
                    else row["surface_form"]
                )

            parsed_list.append(row_words)

        return map(" ".join, parsed_list)

    def load_vectorizer(self, path: Path) -> None:
        with path.open(mode="rb") as fp:
            self.vectorizer = pickle.load(fp)

    def load_classifier(self, path: Path) -> None:
        with path.open(mode="rb") as fp:
            self.clf = pickle.load(fp)

    def predict(self, text: str) -> int:
        w = self.wakachi([text])
        vec = self.vectorizer.transform(w)

        return self.clf.predict(vec)
