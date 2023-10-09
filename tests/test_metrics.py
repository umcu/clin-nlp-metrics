import json
import pickle

import pytest

from clin_nlp_metrics import Dataset, Metrics


@pytest.fixture
def mctrainer_dataset():
    with open("tests/data/medcattrainer_export.json", "rb") as f:
        mctrainer_data = json.load(f)

    return Dataset.from_medcattrainer(data=mctrainer_data)


@pytest.fixture
def clinlp_dataset():
    with open("tests/data/clinlp_docs.pickle", "rb") as f:
        data = pickle.load(f)

    ids = list(f"doc_{x}" for x in range(0, 15))

    return Dataset.from_clinlp_docs(nlp_docs=data, ids=ids)


class TestMetrics:
    def test_entity_metrics(self, mctrainer_dataset, clinlp_dataset):
        nlp_metrics = Metrics(mctrainer_dataset, clinlp_dataset)

        metrics = nlp_metrics.entity_metrics()

        assert list(metrics.keys()) == ["ent_type", "partial", "strict", "exact"]
        assert metrics["strict"]["actual"] == 11
        assert metrics["strict"]["correct"] == 10
        assert metrics["strict"]["precision"] == 0.9090909090909091
        assert metrics["strict"]["recall"] == 0.7692307692307693
        assert metrics["strict"]["f1"] == 0.8333333333333333

    def test_entity_metrics_filter(self, mctrainer_dataset, clinlp_dataset):
        def filter_default(ann):
            return all(qualifier["is_default"] for qualifier in ann.qualifiers)

        nlp_metrics = Metrics(mctrainer_dataset, clinlp_dataset)
        metrics = nlp_metrics.entity_metrics(ann_filter=filter_default)

        assert metrics["strict"]["actual"] == 6
        assert metrics["strict"]["correct"] == 4
        assert metrics["strict"]["precision"] == 0.6666666666666666
        assert metrics["strict"]["recall"] == 0.5
        assert metrics["strict"]["f1"] == 0.5714285714285715

    def test_entity_metrics_classes(self, mctrainer_dataset, clinlp_dataset):
        nlp_metrics = Metrics(mctrainer_dataset, clinlp_dataset)

        metrics = nlp_metrics.entity_metrics(classes=True)

        assert len(metrics) == 9
        assert metrics["C0151526_prematuriteit"]["strict"]["actual"] == 2
        assert metrics["C0151526_prematuriteit"]["strict"]["correct"] == 1
        assert metrics["C0151526_prematuriteit"]["strict"]["precision"] == 0.5
        assert metrics["C0151526_prematuriteit"]["strict"]["recall"] == 0.5
        assert metrics["C0151526_prematuriteit"]["strict"]["f1"] == 0.5

    def test_qualifier_metrics(self, mctrainer_dataset, clinlp_dataset):
        nlp_metrics = Metrics(mctrainer_dataset, clinlp_dataset)

        metrics = nlp_metrics.qualifier_metrics()

        assert metrics["Negation"]["metrics"] == {
            "n": 10,
            "n_pos_pred": 2,
            "n_pos_true": 2,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
        assert metrics["Experiencer"]["metrics"] == {
            "n": 10,
            "n_pos_pred": 1,
            "n_pos_true": 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
        assert metrics["Plausibility"]["metrics"] == {
            "n": 10,
            "n_pos_pred": 2,
            "n_pos_true": 2,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        }
        assert metrics["Temporality"]["metrics"] == {
            "n": 10,
            "n_pos_pred": 1,
            "n_pos_true": 2,
            "precision": 1.0,
            "recall": 0.5,
            "f1": 0.6666666666666666,
        }

    def test_qualifier_misses(self, mctrainer_dataset, clinlp_dataset):
        nlp_metrics = Metrics(mctrainer_dataset, clinlp_dataset)

        metrics = nlp_metrics.qualifier_metrics()

        assert len(metrics["Negation"]["misses"]) == 0
        assert len(metrics["Experiencer"]["misses"]) == 0
        assert len(metrics["Plausibility"]["misses"]) == 2
        assert len(metrics["Temporality"]["misses"]) == 1

    def test_create_metrics_unequal_length(self, mctrainer_dataset, clinlp_dataset):
        mctrainer_dataset.docs = mctrainer_dataset.docs[:-2]

        with pytest.raises(ValueError):
            _ = Metrics(mctrainer_dataset, clinlp_dataset)

    def test_create_metrics_unequal_names(self, mctrainer_dataset, clinlp_dataset):
        mctrainer_dataset.docs[0].identifier = "test"

        with pytest.raises(ValueError):
            _ = Metrics(mctrainer_dataset, clinlp_dataset)
