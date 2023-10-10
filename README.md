# clin-nlp-metrics

This package is intended to make evaluation of clinical nlp algorithms easier, by creating standard methods for evaluating entity matching. It's still in early phases of development. 

## Installation

To install the clin-nlp-metrics package use:

```{bash}
pip install -e .
```

## Usage

### Creating `Dataset`

A small example to create `Dataset` objects, which can be used for computing stats and metrics:

```python
from clin_nlp_metrics import Dataset
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

### Descriptive statistics

Get descriptive statistics for a `Dataset` as follows:

```python
d1.stats()
```

Resulting in:

```python
{'num_docs': 50,
 'num_annotations': 513,
 'span_counts': {'prematuriteit': 43,
                 'infectie': 31,
                 'fototherapie': 25,
                 'dysmaturiteit': 24,
                 'IRDS': 20,
                 'prematuur': 15,
                 'sepsis': 15,
                 'hyperbilirubinemie': 14,
                 'Prematuriteit': 14,
                 'ROP': 13,
                 'necrotiserende enterocolitis': 12,
                 'Prematuur': 11,
                 'infektie': 11,
                 'ductus': 11,
                 'bloeding': 8,
                 'dysmatuur': 7,
                 'IUGR': 7,
                 'Hyperbilirubinemie': 7,
                 'transfusie': 6,
                 'hyperbilirubinaemie': 6,
                 'Dopamine': 6,
                 'wisseltransfusie': 5,
                 'premature partus': 5,
                 'retinopathy of prematurity': 5,
                 'bloedtransfusie': 5},
 'label_counts': {'C0151526_prematuriteit': 94,
                  'C0020433_hyperbilirubinemie': 68,
                  'C0243026_sepsis': 63,
                  'C0015934_intrauterine_groeivertraging': 57,
                  'C0002871_anemie': 37,
                  'C0035220_infant_respiratory_distress_syndrome': 25,
                  'C0035344_retinopathie_van_de_prematuriteit': 21,
                  'C0520459_necrotiserende_enterocolitis': 18,
                  'C0013274_patent_ductus_arteriosus': 18,
                  'C0020649_hypotensie': 18,
                  'C0559477_perinatale_asfyxie': 18,
                  'C0270191_intraventriculaire_bloeding': 17,
                  'C0877064_post_hemorrhagische_ventrikeldilatatie': 13,
                  'C0014850_oesophagus_atresie': 12,
                  'C0006287_bronchopulmonale_dysplasie': 9,
                  'C0031190_persisterende_pulmonale_hypertensie': 7,
                  'C0015938_macrosomie': 6,
                  'C0751954_veneus_infarct': 5,
                  'C0025289_meningitis': 5,
                  'C0023529_periventriculaire_leucomalacie': 2},
 'qualifier_counts': {'Negation': {'Affirmed': 450, 'Negated': 50},
                      'Plausibility': {'Plausible': 452, 'Hypothetical': 48},
                      'Temporality': {'Current': 482, 'Historical': 18},
                      'Experiencer': {'Patient': 489, 'Other': 11}}}
```

### Metrics

Create a `Metrics` object as follows:

```python
from clin_nlp_metrics import Metrics

nlp_metrics = Metrics(d1, d2)

nlp_metrics.entity_metrics()
```

Will result in:
```python
{'ent_type': {'correct': 480,
              'incorrect': 1,
              'partial': 0,
              'missed': 32,
              'spurious': 21,
              'possible': 513,
              'actual': 502,
              'precision': 0.9561752988047809,
              'recall': 0.935672514619883,
              'f1': 0.9458128078817734},
 'partial': {'correct': 473,
             'incorrect': 0,
             'partial': 8,
             'missed': 32,
             'spurious': 21,
             'possible': 513,
             'actual': 502,
             'precision': 0.950199203187251,
             'recall': 0.9298245614035088,
             'f1': 0.9399014778325123},
 'strict': {'correct': 473,
            'incorrect': 8,
            'partial': 0,
            'missed': 32,
            'spurious': 21,
            'possible': 513,
            'actual': 502,
            'precision': 0.9422310756972112,
            'recall': 0.9220272904483431,
            'f1': 0.9320197044334976},
 'exact': {'correct': 473,
           'incorrect': 8,
           'partial': 0,
           'missed': 32,
           'spurious': 21,
           'possible': 513,
           'actual': 502,
           'precision': 0.9422310756972112,
           'recall': 0.9220272904483431,
           'f1': 0.9320197044334976}}
```

For explanation on the different metrics (`partial`, `exact`, `strict` and `ent_type`), see [Nervaluate documentation](https://github.com/MantisAI/nervaluate).

Then, for metrics on qualifiers, use:

```python
nlp_metrics.qualifier_info()
```

Resulting in:

```python
{'Experiencer': {'metrics': {'n': 460,
                             'precision': 0.3333333333333333,
                             'recall': 0.09090909090909091,
                             'f1': 0.14285714285714288},
                 'misses': [{'doc.identifier': 'doc_0001',
                             'annotation': {'text': 'anemie',
                                            'start': 1849,
                                            'end': 1855,
                                            'label': 'C0002871_anemie'},
                             'true_qualifier': 'Other',
                             'pred_qualifier': 'Patient'}, ...]},
 'Temporality': {'metrics': {'n': 460,
                             'precision': 0.0,
                             'recall': 0.0,
                             'f1': 0.0},
                 'misses': [{'doc.identifier': 'doc_0001',
                             'annotation': {'text': 'premature partus',
                                            'start': 1611,
                                            'end': 1627,
                                            'label': 'C0151526_prematuriteit'},
                             'true_qualifier': 'Current',
                             'pred_qualifier': 'Historical'}, ...]},
 'Plausibility': {'metrics': {'n': 460,
                              'precision': 0.6486486486486487,
                              'recall': 0.5217391304347826,
                              'f1': 0.5783132530120482},
                  'misses': [{'doc.identifier': 'doc_0001',
                              'annotation': {'text': 'Groeivertraging',
                                             'start': 1668,
                                             'end': 1683,
                                             'label': 'C0015934_intrauterine_groeivertraging'},
                              'true_qualifier': 'Plausible',
                              'pred_qualifier': 'Hypothetical'}, ...]},
 'Negation': {'metrics': {'n': 460,
                          'precision': 0.7692307692307693,
                          'recall': 0.6122448979591837,
                          'f1': 0.6818181818181818},
              'misses': [{'doc.identifier': 'doc_0001',
                          'annotation': {'text': 'wisseltransfusie',
                                         'start': 4095,
                                         'end': 4111,
                                         'label': 'C0020433_hyperbilirubinemie'},
                          'true_qualifier': 'Affirmed',
                          'pred_qualifier': 'Negated'}, ...]}}
```

For some more advanced settings, please refer to the docs/docstrings.

## Documentation
Generate the Sphinx documentation as follows:

```
sphinx-build -b html docs docs/_build
```
## Authors
  * Richard Bartels (r.t.bartels-6@umcutrecht.nl)
  * Vincent Menger (v.j.menger-2@umcutrecht.nl)
  * Ruben Peters (r.peters-7@umcutrecht.nl)
