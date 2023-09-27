import itertools
from dataclasses import dataclass
from typing import Iterable, Optional

from nervaluate import Evaluator
import spacy


@dataclass
class Annotation:
    """
    An annotation of a single entity in a piece of text.
    """

    text: str
    """ The text/str span of this annotation """

    start: int
    """ The start char """

    end: int
    """ The end char"""

    label: str
    """ The label/tag"""

    qualifiers: Optional[list[str]] = None
    """ Optionally, a list of qualifiers"""

    def to_nervaluate(self) -> dict:
        """
        Converts to format that nervaluate ingests.

        Returns
        -------
        A dictionary with the items nervaluate expects.

        """
        return {"start": self.start, "end": self.end, "label": self.label}


@dataclass
class Document:
    """ A document corresponds to any piece of text that is annotated. """

    identifier: str
    """ Any identifier for the document. """

    text: str
    """ The text. """

    annotations: list[Annotation]
    """ A list of annotations. """

    def to_nervaluate(self) -> list[dict]:
        """
        Converts to format that nervaluate ingests.

        Returns
        -------
        A list of dictionaries corresponding to annotations.

        """
        return [ann.to_nervaluate() for ann in self.annotations]


@dataclass
class Dataset:
    """ A dataset consists of a number of annotated documents. """

    docs: list[Document]
    """ The annotated documents. """

    @staticmethod
    def from_clinlp_docs(
        nlp_docs: Iterable[spacy.language.Doc], ids: Optional[Iterable[str]] = None
    ) -> "Dataset":
        """
        Creates a new dataset from clinlp output, by converting the spaCy Docs.

        Parameters
        ----------
        nlp_docs: An iterable of docs produced by clinlp (a generator from nlp.pipe also works)
        ids: An iterable of identifiers, that should have the same length as nlp_docs. If none is provided,
        a simple counter will be used.

        Returns
        -------
        A Dataset, corresponding to the provided spaCy docs that clinlp produced.
        """
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
                Document(identifier=str(identifier), text=doc.text, annotations=annotations)
            )

        return Dataset(docs)

    @staticmethod
    def from_medcattrainer(data: dict) -> "Dataset":
        """
        Creates a new dataset from medcattrainer output, by converting downloaded json.

        Parameters
        ----------
        data: The output from medcattrainer, as downloaded from the interface in json format and provided as a dict.

        Returns
        -------
        A Dataset, corresponding to the provided data that medcattrainer produced.

        """

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
        """
        Converts to format that nervaluate ingests.

        Returns
        -------
        A nested list of dictionaries corresponding to annotations.
        """

        return [doc.to_nervaluate() for doc in self.docs]


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