from collections import defaultdict

from nervaluate import Evaluator
from sklearn.metrics import f1_score, precision_score, recall_score

from clin_nlp_metrics.dataset import Dataset


class Metrics:
    _QUALIFIER_METRICS = {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }

    def __init__(self, true: Dataset, pred: Dataset):
        self.true = true
        self.pred = pred

        self._validate()

    def _validate(self):
        if self.true.num_docs() != self.pred.num_docs():
            raise ValueError("Can only compute metrics for Datasets with same size")

        for true_doc, pred_doc in zip(self.true.docs, self.pred.docs):
            if true_doc.identifier != pred_doc.identifier:
                raise ValueError(
                    f"Found two documents with non-matching ids "
                    f"(true={true_doc.identifier}, pred={pred_doc.identifier}). "
                    f"Please make sure to present the same documents, "
                    f"in the same order."
                )

    def entities(self, classes=False) -> dict:
        true = self.true.to_nervaluate()
        pred = self.pred.to_nervaluate()

        labels = self.true.labels.union(self.pred.labels)

        evaluator = Evaluator(true=true, pred=pred, tags=labels)
        results, full_results = evaluator.evaluate()

        if classes:
            return full_results
        else:
            return results

    @classmethod
    def _compute_metrics(cls, qualifier_values: dict[str, dict[str, list]]):
        metrics = {}

        _DEFAULTS = {
            "Negation": "Negated",
            "Temporality": "Historical",
            "Experiencer": "Other",
            "Plausibility": "Hypothetical",
        }

        for name, values in qualifier_values.items():
            metrics[name] = {'n': len(values["true"])}

            for metric_name, metric in cls._QUALIFIER_METRICS.items():
                metrics[name][metric_name] = metric(
                    values["true"], values["pred"], pos_label=_DEFAULTS[name]
                )

        return metrics

    def qualifiers(self) -> dict:
        qualifier_values = defaultdict(lambda: defaultdict(list))
        misses = defaultdict(list)

        for true_doc, pred_doc in zip(self.true.docs, self.pred.docs):
            for true_annotation in true_doc.annotations:
                pred_annotation = pred_doc.get_annotation_from_span(
                    start=true_annotation.start, end=true_annotation.end
                )

                if pred_annotation is None:
                    continue

                qualifier_names = true_annotation.qualifier_names.intersection(
                    pred_annotation.qualifier_names
                )

                for name in qualifier_names:
                    true_val = true_annotation.get_qualifier_by_name(name)["value"]
                    pred_val = pred_annotation.get_qualifier_by_name(name)["value"]

                    qualifier_values[name]["true"].append(true_val)
                    qualifier_values[name]["pred"].append(pred_val)

                    if true_val != pred_val:
                        misses[name].append(
                            {
                                "doc.identifier": true_doc.identifier,
                                "annotation": {
                                    "text": true_annotation.text,
                                    "start": true_annotation.start,
                                    "end": true_annotation.end,
                                },
                                "qualifier": name,
                                "true_qualifier": true_val,
                                "pred_qualifier": pred_val,
                            }
                        )

        return {"metrics": self._compute_metrics(qualifier_values), "misses": misses}
