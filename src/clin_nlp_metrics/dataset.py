import inspect
import itertools
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import spacy

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


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

    qualifiers: list[dict] = field(default_factory=lambda: list())
    """ Optionally, a list of qualifiers"""

    def lstrip(self, chars=" ,"):
        """
        Strips punctuation and whitespaces from the beginning of the annotation.
        """

        self.start += len(self.text) - len(self.text.lstrip(chars))
        self.text = self.text.lstrip(chars)

    def rstrip(self, chars=" ,"):
        """
        Strips punctuation and whitespaces from the end of the annotation.
        """

        self.end -= len(self.text) - len(self.text.rstrip(chars))
        self.text = self.text.rstrip(chars)

    def strip(self, chars=" ,"):
        """
        Strips punctuation and whitespaces from the beginning and end of the annotation.
        """

        self.lstrip(chars=chars)
        self.rstrip(chars=chars)

    def to_nervaluate(self) -> dict:
        """
        Converts to format that nervaluate ingests.

        Returns
        -------
        A dictionary with the items nervaluate expects.
        """
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "label": self.label,
        }

    @property
    def qualifier_names(self) -> set[str]:
        """
        Obtain unique qualifier names for this annotation.

        Returns
        -------
        A set of unique qualifier names, e.g. {"Negation", "Experiencer"}.
        """
        return {qualifier["name"] for qualifier in self.qualifiers}

    def get_qualifier_by_name(self, qualifier_name: str) -> dict:
        """
        Get a qualifier from the set of qualifiers by its name, or raise an error
        when this qualifier is not present.

        Parameters
        ----------
        qualifier_name: The name of the qualifier, e.g. Negation

        Returns
        -------
        The entire qualifier, e.g. {"name": "Negation", "value": "Affirmed", ...}

        """
        for qualifier in self.qualifiers:
            if qualifier["name"] == qualifier_name:
                return qualifier

        raise KeyError(f"No qualifier with name {qualifier_name}.")


@dataclass
class Document:
    """A document corresponds to any piece of text that is annotated."""

    identifier: str
    """ Any identifier for the document. """

    text: str
    """ The text. """

    annotations: list[Annotation]
    """ A list of annotations. """

    def to_nervaluate(
        self, ann_filter: Optional[Callable[[Annotation], bool]] = None
    ) -> list[dict]:
        """
        Converts to format that nervaluate ingests.

        Parameters
        ----------
        ann_filter: A filter to apply to annotations, should map to annotations to True
        if they should be included, False otherwise.

        Returns
        -------
        A list of dictionaries corresponding to annotations.
        """

        ann_filter = ann_filter or (lambda ann: True)

        return [ann.to_nervaluate() for ann in self.annotations if ann_filter(ann)]

    def labels(
        self, ann_filter: Optional[Callable[[Annotation], bool]] = None
    ) -> set[str]:
        """
        Obtain all annotation labels for this document.

        Parameters
        ----------
        ann_filter: A filter to apply to annotations, should map to annotations to True
        if they should be included, False otherwise.

        Returns
        -------
        A set containing all annotation labels for this document.
        """
        ann_filter = ann_filter or (lambda ann: True)

        return {
            annotation.label
            for annotation in self.annotations
            if ann_filter(annotation)
        }

    def get_annotation_from_span(self, start: int, end: int) -> Optional[Annotation]:
        """
        Get annotation that exactly matches start and end char.

        Parameters
        ----------
        start: The start char.
        end: The end char.

        Returns
        -------
        The Annotation with the provided start and end char, of None if no such
        Annotation exists.
        """

        for annotation in self.annotations:
            if (annotation.start == start) and (annotation.end == end):
                return annotation

        return None


@dataclass
class Dataset:
    """A dataset consists of a number of annotated documents."""

    docs: list[Document]
    """ The annotated documents. """

    default_qualifiers: dict[str, str] = None
    """ Mapping of qualifiers to their default value, e.g. {"Negation": "Affirmed"}"""

    def __post_init__(self):
        """
        Responsible for setting the default qualifiers by checking the default
        qualifiers set on Annotations, if set, or inferring them from the majority
        class otherwise.
        """

        self.default_qualifiers = {}

        try:
            for doc in self.docs:
                for annotation in doc.annotations:
                    for qualifier in annotation.qualifiers:
                        if qualifier["is_default"]:
                            self.default_qualifiers[qualifier["name"]] = qualifier[
                                "value"
                            ]
        except KeyError:
            self.default_qualifiers = self.infer_default_qualifiers()

    _ALL_STATS = [
        "num_docs",
        "num_annotations",
        "span_counts",
        "label_counts",
        "qualifier_counts",
    ]
    """ All methods to call when computing full dataset stats """

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

        return Dataset(docs=docs)

    def infer_default_qualifiers(self) -> dict:
        """
        Infer the default values for qualifiers, based on the majority class, and
        set this value on all annotations.

        Returns
        A dictionary with defaults, e.g. {"Negation": "Negated", "Experiencer":
        "Patient"}.
        -------
        """

        default_qualifiers = {
            name: max(counts, key=lambda item: counts[item])
            for name, counts in self.qualifier_counts().items()
        }

        logger.warning(
            f"Inferred the following qualifier defaults from the majority "
            f"classes: {default_qualifiers}. "
        )

        for doc in self.docs:
            for annotation in doc.annotations:
                for qualifier in annotation.qualifiers:
                    qualifier["is_default"] = (
                        default_qualifiers[qualifier["name"]] == qualifier["value"]
                    )

        return default_qualifiers

    @staticmethod
    def from_medcattrainer(
        data: dict,
        strip_spans: bool = True,
        default_qualifiers: Optional[dict[str, str]] = None,
    ) -> "Dataset":
        """
        Creates a new dataset from medcattrainer output, by converting downloaded json.

        Parameters
        ----------
        data: The output from medcattrainer, as downloaded from the interface in
        json format and provided as a dict.
        strip_spans: Whether to remove punctuation and whitespaces from the beginning or
        end of annotations. Used to clean up accidental over-annotations.
        default_qualifiers: Optionally, the default qualifiers (which are not included
        in the medcattrainer export), e.g. {"Negation": "Negated", "Experiencer":
        "Patient"}. If None, will assume majority class is default.

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
                        qualifier = {
                            "name": qualifier["name"].title(),
                            "value": qualifier["value"].title(),
                        }

                        if default_qualifiers is not None:
                            qualifier["is_default"] = (
                                default_qualifiers[qualifier["name"]]
                                == qualifier["value"]
                            )

                        qualifiers.append(qualifier)

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

    def to_nervaluate(
        self, ann_filter: Optional[Callable[[Annotation], bool]] = None
    ) -> list[list[dict]]:
        """
        Converts to format that nervaluate ingests.

        Parameters
        ----------
        ann_filter: A filter to apply to annotations, should map to annotations to True
        if they should be included, False otherwise.

        Returns
        -------
        A nested list of dictionaries corresponding to annotations.
        """

        ann_filter = ann_filter or (lambda ann: True)

        return [doc.to_nervaluate(ann_filter) for doc in self.docs]

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
        self,
        n_spans: Optional[int] = 25,
        span_callback: Optional[Callable] = None,
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
        self,
        n_labels: Optional[int] = 25,
        label_callback: Optional[Callable] = None,
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

    def stats(self, **kwargs) -> dict:
        """
        Compute all the stats of this dataset, as defined in the
        _ALL_STATS class variable.

        Returns
        -------
        A dictionary mapping the name of the stat to the values.
        E.g.: {'num_docs': 384, 'num_annotations': 4353, ...}
        """

        stats = {}

        for stat in self._ALL_STATS:
            stat_func = getattr(self, stat)

            func_kwargs = {
                k: kwargs[k]
                for k in inspect.signature(stat_func).parameters
                if k in kwargs
            }

            stats[stat] = stat_func(**func_kwargs)

        return stats
