from collections import defaultdict
from typing import Callable, Optional

import nervaluate
from sklearn.metrics import f1_score, precision_score, recall_score

from clin_nlp_metrics.dataset import Annotation, Dataset

""" Compute these metrics for qualifiers. """
_QUALIFIER_METRICS = {
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

class Metrics:
    """
    Use this class to implement metrics comparing datasets.
    """

    def __init__(self, true: Dataset, pred: Dataset):
        """
        Initialize metrics.

        Parameters
        ----------
        true: The dataset containing true (annotated/gold standard) annotations.
        pred: The dataset containing pred (predicted/inferred) annotations.
        """
        self.true = true
        self.pred = pred

        self._validate_self()

    def _validate_self(self):
        """
        Validate the two datasets. Will raise an ValueError when datasets don't contain
        the same documents.
        """
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

    def entity_metrics(
        self,
        ann_filter: Optional[Callable[[Annotation], bool]] = None,
        classes: bool = False,
    ) -> dict:
        """
        Compute metrics for entities, including precision, recall and f1-score.
        Returns all measures for exact, strict, partial, and type matching, based on
        the nervaluate libary (for more information, see:
        https://github.com/MantisAI/nervaluate)

        Parameters
        ----------
        ann_filter: An optional filter to apply to annotations, e.g. including
        only annotations with a certain label or qualifier. Annotations are only
        included if ann_filter evaluates to True.
        classes: Will return metrics per class when set to True, or micro-averaged over
        all classes when set to False.

        Returns
        -------
        A dictionary containing the relevant metrics.

        """

        ann_filter = ann_filter or (lambda ann: True)

        true_anns = self.true.to_nervaluate(ann_filter)
        pred_anns = self.pred.to_nervaluate(ann_filter)

        labels = list(
            set.union(
                *[doc.labels(ann_filter) for doc in self.true.docs + self.pred.docs]
            )
        )

        evaluator = nervaluate.Evaluator(
            true=true_anns, pred=pred_anns, tags=labels, track_ents=True
        )

        results, class_results = evaluator.evaluate()

        return class_results if classes else results

    def _aggregate_qualifier_values(self) -> dict[str, dict[str, list]]:
        """
        Aggregates all Annotation qualifier values for metric computation. Matches
        Annotations from true docs to an Annotation from pred docs with equal span,
        but not necessarily the same label. Only aggregates qualifiers that are present
        in both Annotations.

        Returns
        -------
        For each qualifier, the true values, predicted values, and misses aggregated
        into lists, e.g.:

        {
            "Negation": {
                "true": ["Affirmed", "Negated", "Affirmed"],
                "pred": ["Affirmed", "Negated", "Negated"],
                "misses": [
                    {"doc.identifier": 1, annotation: {"start": 0, "end": 5, "text":
                    "test"}, true_label: "Affirmed", pred_label: "Negated"}, ...]
            },
            ...
        }
        """

        aggregation = defaultdict(lambda: defaultdict(list))

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

                    aggregation[name]["true"].append(true_val)
                    aggregation[name]["pred"].append(pred_val)

                    if true_val != pred_val:
                        aggregation[name]["misses"].append(
                            {
                                "doc.identifier": true_doc.identifier,
                                "annotation": true_annotation.to_nervaluate(),
                                "true_qualifier": true_val,
                                "pred_qualifier": pred_val,
                            }
                        )

        return aggregation

    def qualifier_metrics(self) -> dict:
        """
        Computes metrics for qualifiers, including precision, recall and f1-score.
        Only computes metrics for combinations of annotations with the same start
        and end char, but regardless of whether the labels match.

        Returns
        -------
        A dictionary with metrics for each qualifier.
        """

        aggregation = self._aggregate_qualifier_values()

        result = {}

        for name, values in aggregation.items():
            true_unique_values = set(values["true"])
            pred_unique_values = set(values["pred"])

            if max(len(true_unique_values), len(pred_unique_values)) > 2:
                raise ValueError("Can oly compute metrics for binary qualifier values")

            pos_label = next(
                val
                for val in true_unique_values
                if val != self.true.default_qualifiers[name]
            )

            result[name] = {
                "metrics": {
                    "n": len(values["true"]),
                },
                "misses": values["misses"],
            }

            for metric_name, metric_func in self._QUALIFIER_METRICS.items():
                result[name]["metrics"][metric_name] = metric_func(
                    values["true"], values["pred"], pos_label=pos_label
                )

        return result
