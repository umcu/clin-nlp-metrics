import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

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

    qualifiers: Optional[list[dict]] = None
    """ Optionally, a list of qualifiers"""

    def lstrip(self):
        """
        Strips punctuation and whitespaces from the beginning of the annotation.
        """

        self.start += len(self.text) - len(self.text.lstrip())
        self.text = self.text.lstrip()

    def rstrip(self):
        """
        Strips punctuation and whitespaces from the end of the annotation.
        """

        self.start -= len(self.text) - len(self.text.rstrip())
        self.text = self.text.rstrip()

    def strip(self):
        """
        Strips punctuation and whitespaces from the beginning and end of the annotation.
        """

        self.lstrip()
        self.rstrip()

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
    """A document corresponds to any piece of text that is annotated."""

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
    """A dataset consists of a number of annotated documents."""

    docs: list[Document]
    """ The annotated documents. """

    _ALL_STATS = [
        "num_docs",
        "num_annotations",
        "span_counts",
        "label_counts",
        "qualifier_counts",
    ]
    """ The class methods to call when computing dataset stats """

    @staticmethod
    def from_clinlp_docs(
        nlp_docs: Iterable[spacy.language.Doc], ids: Optional[Iterable[str]] = None
    ) -> "Dataset":
        """
        Creates a new dataset from clinlp output, by converting the spaCy Docs.

        Parameters
        ----------
        nlp_docs: An iterable of docs produced by clinlp (a generator from nlp.pipe
        also works)
        ids: An iterable of identifiers, that should have the same length as nlp_docs.
        If none is provided, a simple counter will be used.

        Returns
        -------
        A Dataset, corresponding to the provided spaCy docs that clinlp produced.
        """
        ids = ids or itertools.count()

        docs = []

        for doc, identifier in zip(nlp_docs, ids):
            annotations = []

            for ent in doc.ents:
                qualifiers = []

                for qualifier in ent._.qualifiers_dict:
                    qualifiers.append(
                        {
                            "name": qualifier["name"].title(),
                            "value": qualifier["value"].title(),
                            "is_default": qualifier["is_default"],
                        }
                    )

                annotations.append(
                    Annotation(
                        text=str(ent),
                        start=ent.start_char,
                        end=ent.end_char,
                        label=ent.label_,
                        qualifiers=qualifiers,
                    )
                )

            docs.append(
                Document(
                    identifier=str(identifier), text=doc.text, annotations=annotations
                )
            )

        return Dataset(docs)

    @staticmethod
    def from_medcattrainer(data: dict, strip_spans: bool = True) -> "Dataset":
        """
        Creates a new dataset from medcattrainer output, by converting downloaded json.

        Parameters
        ----------
        data: The output from medcattrainer, as downloaded from the interface in
        json format and provided as a dict.
        strip_spans: Whether to remove punctuation and whitespaces from the beginning or
        end of annotations. Used to clean up accidental over-annotations.

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
                    qualifiers = []

                    for qualifier in annotation["meta_anns"].values():
                        qualifiers.append(
                            {
                                "name": qualifier["name"].title(),
                                "value": qualifier["value"].title(),
                            }
                        )

                    annotation = Annotation(
                        text=annotation["value"],
                        start=annotation["start"],
                        end=annotation["end"],
                        label=annotation["cui"],
                        qualifiers=qualifiers,
                    )

                    if strip_spans:
                        annotation.strip()

                    annotations.append(annotation)

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

    def num_docs(self) -> int:
        """
        The number of documents in this dataset.

        Returns
        -------
        The number of documents in this dataset.

        """
        return len(self.docs)

    def num_annotations(self) -> int:
        """
        The number of annotations in all documents of this dataset.

        Returns
        -------
        The number of annotations in all documents of this dataset.

        """
        return sum(len(doc.annotations) for doc in self.docs)

    def span_counts(
        self, n_spans: Optional[int] = 25, span_callback: Optional[Callable] = None, **kwargs
    ) -> dict:
        """
        Counts the text spans of all annotations in this dataset.

        Parameters
        ----------
        n_spans: The maximum number of spans to return, ordered by frequency
        span_callback: A callback that is applied to each text span

        Returns
        -------
        A dictionary containing the frequency of the requested text spans.
        """
        cntr = Counter()
        span_callback = span_callback or (lambda x: x)

        for doc in self.docs:
            cntr.update(
                [span_callback(annotation.text) for annotation in doc.annotations]
            )

        if n_spans is None:
            n_spans = len(cntr)

        return dict(cntr.most_common(n_spans))

    def label_counts(
        self, n_labels: Optional[int] = 25, label_callback: Optional[Callable] = None, **kwargs
    ) -> dict:
        """
        Counts the annotation labels of all annotations in this dataset.

        Parameters
        ----------
        n_labels: The maximum number of labels to return, ordered by frequency
        label_callback: A callback that is applied to each label

        Returns
        -------
        A dictionary containing the frequency of the requested annotation labels.
        """

        cntr = Counter()
        label_callback = label_callback or (lambda x: x)

        for doc in self.docs:
            cntr.update(
                [label_callback(annotation.label) for annotation in doc.annotations]
            )

        if n_labels is None:
            n_labels = len(cntr)

        return dict(cntr.most_common(n_labels))

    def qualifier_counts(self) -> dict:
        """
        Counts the values of all qualifiers.

        Returns
        -------
        A dictionary, mapping qualifier names to the frequencies of their values.
        E.g.: {"Negation": {"Affirmed": 34, "Negated": 12}}

        """
        cntrs = defaultdict(lambda: Counter())

        for doc in self.docs:
            for annotation in doc.annotations:
                for qualifier in annotation.qualifiers:
                    cntrs[qualifier["name"]].update([qualifier["value"]])

        return {name: dict(counts) for name, counts in cntrs.items()}

    def get_stats(self, **kwargs):
        return {stat: getattr(self, stat)(**kwargs) for stat in self._ALL_STATS}
