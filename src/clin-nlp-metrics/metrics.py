from dataclasses import dataclass
from typing import Iterable, Optional

import spacy
from nervaluate import Evaluator


@dataclass
class Annotation:
    text: str
    start: int
    end: int
    label: str
    qualifiers: Optional[list[str]] = None

    def to_nervaluate(self) -> dict:
        return {"start": self.start, "end": self.end, "label": self.label}


@dataclass
class Document:
    identifier: str
    text: str
    annotations: list[Annotation]


@dataclass
class Dataset:
    docs: list[Document]

    @staticmethod
    def from_clinlp_docs(
        nlp_docs: Iterable[spacy.language.Doc], ids: Optional[Iterable[str]] = None
    ) -> "Dataset":
        ids = ids or itertools.count()

        docs = []

        for doc, identifier in zip(nlp_docs, ids):
            annotations = []

            for ent in doc.ents:
                annotations.append(
                    Annotation(
                        text=str(ent),
                        start=ent.start_char,
                        end=ent.end_char,
                        label=ent.label_,
                    )
                )

            docs.append(
                Document(identifier=identifier, text=doc.text, annotations=annotations)
            )

        return Dataset(docs)

    @staticmethod
    def from_medcattrainer(data: dict) -> "Dataset":
        if len(data["projects"]) > 1:
            raise ValueError(
                "Cannot read MedCATTrainer exports with more than 1 project."
            )

        data = data["projects"][0]
        docs = []

        for doc in data["documents"]:
            annotations = []

            for annotation in doc["annotations"]:
                if not annotation["deleted"]:
                    annotations.append(
                        Annotation(
                            text=annotation["value"],
                            start=annotation["start"],
                            end=annotation["end"],
                            label=annotation["cui"],
                        )
                    )

            docs.append(
                Document(
                    identifier=doc["name"], text=doc["text"], annotations=annotations
                )
            )

        return Dataset(docs)

    def to_nervaluate(self) -> list[list[dict]]:
        return [
            list(ann.to_nervaluate() for ann in doc.annotations) for doc in self.docs
        ]


# Example metric 1
def stats(data: Dataset) -> dict:
    results = {
        "num_docs": len(data.docs),
        "num_annotations": sum(len(doc.annotations) for doc in data.docs),
    }

    return results


# Example metric 2
def entity_metrics(true: Dataset, pred: Dataset) -> dict:
    true = true.to_nervaluate()
    pred = pred.to_nervaluate()

    tags = list({annotation["label"] for doc in true + pred for annotation in doc})
    evaluator = Evaluator(true=true, pred=pred, tags=tags)
    results, _ = evaluator.evaluate()

    return results
