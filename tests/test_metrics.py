import json
import pickle

import clinlp  # noqa: F401
import pytest

from clin_nlp_metrics.metrics import Annotation, Dataset, Document


@pytest.fixture
def dataset():
    with open("tests/data/medcattrainer_export.json", "rb") as f:
        mctrainer_data = json.load(f)

    return Dataset.from_medcattrainer(data=mctrainer_data)


class TestAnnotation:
    def test_annotation_nervaluate(self):
        ann = Annotation(text="test", start=0, end=5, label="test")

        assert ann.to_nervaluate() == {"start": 0, "end": 5, "label": "test"}

    def test_annotation_lstrip(self):
        ann = Annotation(text=" test", start=0, end=5, label="test")

        ann.lstrip()

        assert ann == Annotation(text="test", start=1, end=5, label="test")

    def test_annotation_rstrip(self):
        ann = Annotation(text="test,", start=0, end=5, label="test")

        ann.rstrip()

        assert ann == Annotation(text="test", start=0, end=4, label="test")

    def test_annotation_strip(self):
        ann = Annotation(text=" test,", start=0, end=6, label="test")

        ann.strip()

        assert ann == Annotation(text="test", start=1, end=5, label="test")


class TestDocument:
    def test_document_nervaluate(self):
        doc = Document(
            identifier="1",
            text="test1 and test2",
            annotations=[
                Annotation(text="test1", start=0, end=5, label="test1"),
                Annotation(text="test2", start=10, end=15, label="test2"),
            ],
        )

        assert doc.to_nervaluate() == [
            {"start": 0, "end": 5, "label": "test1"},
            {"start": 10, "end": 15, "label": "test2"},
        ]


class TestDataset:
    def test_dataset_from_medcattrainer(self):
        with open("tests/data/medcattrainer_export.json", "rb") as f:
            mctrainer_data = json.load(f)

        dataset = Dataset.from_medcattrainer(data=mctrainer_data)

        assert len(dataset.docs) == 2
        assert dataset.docs[0].text == "random text sample"
        assert len(dataset.docs[0].annotations) == 1
        assert len(dataset.docs[1].annotations) == 3

        assert dataset.docs[0].annotations[0].text == "anemie"
        assert dataset.docs[0].annotations[0].start == 978
        assert dataset.docs[0].annotations[0].end == 984
        assert dataset.docs[0].annotations[0].label == "C0002871_anemie"

        assert dataset.docs[1].annotations[0].text == "<< p3"
        assert dataset.docs[1].annotations[0].start == 1739
        assert dataset.docs[1].annotations[0].end == 1744
        assert (
            dataset.docs[1].annotations[0].label
            == "C0015934_intrauterine_groeivertraging"
        )

        assert dataset.docs[0].annotations[0].qualifiers == [
            {"name": "Plausibility", "value": "Plausible"},
            {"name": "Temporality", "value": "Current"},
            {"name": "Negation", "value": "Negated"},
            {"name": "Experiencer", "value": "Patient"},
        ]

    def test_dataset_from_clinlp(self):
        with open("tests/data/clinlp_docs.pickle", "rb") as f:
            clinlp_docs = pickle.load(f)

        dataset = Dataset.from_clinlp_docs(nlp_docs=clinlp_docs)

        assert len(dataset.docs) == 3
        assert dataset.docs[0].text == "patient had geen anemie"
        assert len(dataset.docs[0].annotations) == 1
        assert len(dataset.docs[1].annotations) == 2
        assert len(dataset.docs[2].annotations) == 1

        assert dataset.docs[0].annotations[0].text == "anemie"
        assert dataset.docs[0].annotations[0].start == 17
        assert dataset.docs[0].annotations[0].end == 23
        assert dataset.docs[0].annotations[0].label == "C0002871_anemie"

        assert dataset.docs[1].annotations[0].text == "prematuriteit"
        assert dataset.docs[1].annotations[0].start == 18
        assert dataset.docs[1].annotations[0].end == 31
        assert dataset.docs[1].annotations[0].label == "C0151526_prematuriteit"

        assert sorted(
            dataset.docs[0].annotations[0].qualifiers, key=lambda q: q["name"]
        ) == [
            {"name": "Experiencer", "value": "Patient", "is_default": True},
            {"name": "Negation", "value": "Negated", "is_default": False},
            {"name": "Plausibility", "value": "Plausible", "is_default": True},
            {"name": "Temporality", "value": "Current", "is_default": True},
        ]

    def test_dataset_nervaluate(self):
        dataset = Dataset(
            docs=[
                Document(
                    identifier="1",
                    text="test1",
                    annotations=[
                        Annotation(
                            text="test1",
                            start=0,
                            end=5,
                            label="test1",
                            qualifiers=[{"name": "Negation", "value": "Negated"}],
                        ),
                    ],
                ),
                Document(
                    identifier="2",
                    text="test2",
                    annotations=[
                        Annotation(text="test2", start=0, end=5, label="test2"),
                    ],
                ),
            ]
        )

        assert dataset.to_nervaluate() == [
            [{"start": 0, "end": 5, "label": "test1"}],
            [{"start": 0, "end": 5, "label": "test2"}],
        ]

    def test_num_docs(self, dataset):
        assert dataset.num_docs() == 2

    def test_num_annotations(self, dataset):
        assert dataset.num_annotations() == 4

    def test_span_counts(self, dataset):
        assert dataset.span_counts() == {"<< p3": 1, "<p3": 2, "anemie": 1}

    def test_span_counts_n_spans(self, dataset):
        assert dataset.span_counts(n_spans=1) == {"<p3": 2}

    def test_span_counts_callback(self, dataset):
        assert dataset.span_counts(span_callback=lambda x: x.upper()) == {
            "<< P3": 1,
            "<P3": 2,
            "ANEMIE": 1,
        }

    def test_label_counts(self, dataset):
        assert dataset.label_counts() == {
            "C0002871_anemie": 1,
            "C0015934_intrauterine_groeivertraging": 3,
        }

    def test_label_counts_n_labels(self, dataset):
        assert dataset.label_counts(n_labels=1) == {
            "C0015934_intrauterine_groeivertraging": 3
        }

    def test_label_counts_callback(self, dataset):
        assert dataset.label_counts(label_callback=lambda x: x[x.index("_") + 1 :]) == {
            "anemie": 1,
            "intrauterine_groeivertraging": 3,
        }

    def test_qualifier_counts(self, dataset):
        assert dataset.qualifier_counts() == {
            "Experiencer": {"Patient": 4},
            "Negation": {"Affirmed": 3, "Negated": 1},
            "Plausibility": {"Plausible": 4},
            "Temporality": {"Current": 4},
        }

    def test_stats(self, dataset):
        stats = dataset.stats()

        assert stats["num_docs"] == dataset.num_docs()
        assert stats["num_annotations"] == dataset.num_annotations()
        assert stats["span_counts"] == dataset.span_counts()
        assert stats["label_counts"] == dataset.label_counts()
        assert stats["qualifier_counts"] == dataset.qualifier_counts()

    def test_stats_with_kwargs(self, dataset):
        n_labels = 1
        span_callback = lambda x: x.upper()  # noqa: E731

        stats = dataset.stats(
            n_labels=n_labels, span_callback=span_callback, unused_argument=None
        )

        assert stats["num_docs"] == dataset.num_docs()
        assert stats["num_annotations"] == dataset.num_annotations()
        assert stats["span_counts"] == dataset.span_counts(span_callback=span_callback)
        assert stats["label_counts"] == dataset.label_counts(n_labels=n_labels)
        assert stats["qualifier_counts"] == dataset.qualifier_counts()
