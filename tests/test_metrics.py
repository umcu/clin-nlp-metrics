from src.clin_nlp_metrics.metrics import Annotation, Dataset, Document


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
    def test_dataset_nervaluate(self):
        dataset = Dataset(
            docs=[
                Document(
                    identifier="1",
                    text="test1",
                    annotations=[
                        Annotation(text="test1", start=0, end=5, label="test1"),
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
