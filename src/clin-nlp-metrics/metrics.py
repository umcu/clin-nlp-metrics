from dataclasses import dataclass

import spacy

from typing import Iterable

@dataclass
class Annotation:
    text: str
    start: int
    end: int
    label: str
    qualifiers: list[str]


@dataclass
class Document:
    text: str
    annotations: list[Annotation]


@dataclass
class Dataset:
    docs: list[Document]

    @staticmethod
    def from_clinlp_docs(docs: Iterable[spacy.language.Doc]) -> "Dataset":
        pass

    @staticmethod
    def from_medcattrainer(data: dict) -> "Dataset":
        pass

    @staticmethod
    def to_nervaluate() -> dict:
        pass


def metrics(true: Dataset, pred: Dataset) -> dict:
    pass
