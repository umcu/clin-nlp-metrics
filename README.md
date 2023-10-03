# clin-nlp-metrics

This package is intended to make evaluation of clinical nlp algorithms easier, by creating standard methods for evaluating entity matching. It's still in early phases of development. 

## Installation

To install the clin-nlp-metrics package use:

```{bash}
pip install -e .
```

## Usage

A small example to get started:

```python
from clin_nlp_metrics.dataset import Dataset
import json

# medcattrainer
import json

with open('medcattrainer_export.json', 'rb') as f:
    mtrainer_data = json.load(f)

d1 = Dataset.from_medcattrainer(mctrainer_data)

# clinlp
import clinlp
import spacy

from model import get_model  # not included

nlp = get_model()
nlp_docs = nlp.pipe([doc['text'] for doc in data['projects'][0]['documents']])

d2 = Dataset.from_clinlp_docs(nlp_docs)
```

## Documentation
Generate the Sphinx documentation as follows:

```
sphinx-build -b html docs docs/_build
```
## Authors
  * Richard Bartels (r.t.bartels-6@umcutrecht.nl)
  * Vincent Menger (v.j.menger-2@umcutrecht.nl)
  * Ruben Peters (r.peters-7@umcutrecht.nl)

