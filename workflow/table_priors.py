"""Load independently generated, pre-cutoff table forecasts for prediction context."""

import json
import math
from pathlib import Path


FORBIDDEN = {"actual", "target", "truth", "outcome", "abs_error", "covered_90", "baseline_abs_error"}


def scoped_table_priors(path, task_id, entity_ids):
    allowed = set(entity_ids)
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") != task_id or row.get("entity_id") not in allowed:
                continue
            if FORBIDDEN.intersection(row):
                raise ValueError("table prior contains outcome fields")
            entity_id = row["entity_id"]
            if entity_id in result:
                raise ValueError(f"duplicate table prior for {entity_id}")
            prior = {}
            point = row.get("prediction_mean", row.get("mean"))
            if point is not None:
                point = float(point)
                if not math.isfinite(point):
                    raise ValueError(f"nonfinite table prior for {entity_id}")
                prior["point"] = point
            low = row.get("prediction_q05", row.get("q05"))
            high = row.get("prediction_q95", row.get("q95"))
            if low is not None and high is not None:
                low, high = float(low), float(high)
                if not math.isfinite(low) or not math.isfinite(high) or low > high:
                    raise ValueError(f"invalid table interval for {entity_id}")
                prior["interval_90"] = {"lo": low, "hi": high}
            label = row.get("prediction_label")
            if label is not None:
                prior["label"] = str(label)
            probabilities = row.get("probabilities")
            if probabilities is not None:
                prior["probabilities"] = {str(k): float(v) for k, v in probabilities.items()}
            if row.get("train_rows") is not None:
                prior["train_rows"] = int(row["train_rows"])
            if prior:
                result[entity_id] = prior
    return result
