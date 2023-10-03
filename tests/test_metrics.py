import json
import pickle

import clinlp  # noqa: F401

from clin_nlp_metrics.dataset import Annotation, Dataset, Document


class TestAnnotation:
    def test_annotation_nervaluate(self):
        ann = Annotation(text="test", start=0, end=5, label="test")

        assert ann.to_nervaluate() == {"start": 0, "end": 5, "label": "test"}


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
