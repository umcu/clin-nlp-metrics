import json
import pickle

import pytest

from clin_nlp_metrics import Dataset


@pytest.fixture
def mctrainer_data():
    with open("tests/data/medcattrainer_export.json", "rb") as f:
        return json.load(f)


@pytest.fixture
def mctrainer_dataset(mctrainer_data):
    return Dataset.from_medcattrainer(data=mctrainer_data)


@pytest.fixture
def clinlp_docs():
    with open("tests/data/clinlp_docs.pickle", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def clinlp_dataset(clinlp_docs):
    ids = list(f"doc_{x}" for x in range(0, 15))

    return Dataset.from_clinlp_docs(nlp_docs=clinlp_docs, ids=ids)
