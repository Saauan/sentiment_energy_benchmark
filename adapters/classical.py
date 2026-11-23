"""From-scratch classical sentiment classifiers.

``create`` only parses configuration. Dependencies are imported and models are
fitted in ``setup`` so construction has no training or native-runtime side
effects. All three adapters accept static batches of any size.

Configuration schema::

    {
        "adapter": "svm" | "nb" | "fasttext",
        "params": {
            "preprocessing": {...},  # shared TF-IDF options for svm/nb
            "model": {...},          # estimator or fastText training options
            "training": {"seed": 42}
        }
    }

For fastText, preprocessing supports ``lowercase`` and ``strip_accents``;
newlines and other whitespace are always collapsed to one line per example.
Its model options are passed to ``fasttext.train_supervised`` except that
pretrained vectors, autotuning, and label-prefix overrides are deliberately
not supported. The default is a deterministic, from-scratch, single-threaded
model. On CPython 3.13, install ``fasttext==0.9.3`` (a C++11 source build).
On CPython 3.12, ``fasttext-wheel==0.9.2`` is the binary-wheel alternative.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import tempfile
import unicodedata
from typing import Any, Iterable

from .base import Unavailable


LABELS = ("negative", "neutral", "positive")
FASTTEXT_LABEL_PREFIX = "__label__"

_TFIDF_DEFAULTS = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 1.0,
    "max_features": None,
    "lowercase": True,
    "strip_accents": None,
    "token_pattern": r"(?u)\b\w\w+\b",
    "binary": False,
    "norm": "l2",
    "use_idf": True,
    "smooth_idf": True,
    "sublinear_tf": True,
}
_SVM_DEFAULTS = {
    "C": 1.0,
    "penalty": "l2",
    "loss": "squared_hinge",
    "dual": "auto",
    "tol": 1e-4,
    "max_iter": 5000,
    "class_weight": None,
    "random_state": 42,
}
_NB_DEFAULTS = {
    "alpha": 1.0,
    "fit_prior": True,
    "class_prior": None,
}
_FASTTEXT_DEFAULTS = {
    "lr": 0.1,
    "dim": 100,
    "epoch": 5,
    "minCount": 1,
    "wordNgrams": 2,
    "loss": "softmax",
    "bucket": 2_000_000,
    "minn": 0,
    "maxn": 0,
    "thread": 1,
    "seed": 42,
    "verbose": 0,
}
_FASTTEXT_MODEL_OPTIONS = frozenset(
    {
        "lr",
        "dim",
        "ws",
        "epoch",
        "minCount",
        "minCountLabel",
        "minn",
        "maxn",
        "neg",
        "wordNgrams",
        "loss",
        "bucket",
        "thread",
        "lrUpdateRate",
        "t",
        "seed",
        "verbose",
    }
)


def _json_value(value: Any) -> Any:
    """Return a detached, JSON-safe representation of resolved parameters."""

    return json.loads(json.dumps(value))


def _rows_digest(rows: Iterable[tuple[Any, str, int]]) -> str:
    payload = json.dumps(
        [list(row) for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_split(
    rows: Iterable[tuple[Any, str, int]], name: str
) -> list[tuple[Any, str, int]]:
    materialized = list(rows)
    for row in materialized:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ValueError(f"{name} rows must be (row_id, text, integer_gold) tuples")
        _, text, gold = row
        if not isinstance(text, str):
            raise TypeError(f"{name} text values must be strings")
        if type(gold) is not int or gold not in (0, 1, 2):
            raise ValueError(f"{name} gold labels must be integers 0, 1, or 2")
    if name == "train":
        if not materialized:
            raise ValueError("train split must not be empty")
        missing = set(range(3)) - {row[2] for row in materialized}
        if missing:
            raise ValueError(f"train split is missing sentiment classes: {sorted(missing)}")
    return [tuple(row) for row in materialized]


def _split_metadata(
    train: list[tuple[Any, str, int]], validation: list[tuple[Any, str, int]]
) -> dict[str, Any]:
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_class_counts": {
            LABELS[label]: count
            for label, count in sorted(Counter(row[2] for row in train).items())
        },
        "train_sha256": _rows_digest(train),
        "validation_sha256": _rows_digest(validation),
        "validation_used_for_training_or_tuning": False,
    }


def _params(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    params = deepcopy(spec.get("params", {}))
    if not isinstance(params, dict):
        raise TypeError("spec['params'] must be a dict")
    unknown = set(params) - {"preprocessing", "model", "training"}
    if unknown:
        raise ValueError(f"unknown classical parameter sections: {sorted(unknown)}")
    for section in ("preprocessing", "model", "training"):
        value = params.setdefault(section, {})
        if not isinstance(value, dict):
            raise TypeError(f"spec['params']['{section}'] must be a dict")
    unknown_training = set(params["training"]) - {"seed"}
    if unknown_training:
        raise ValueError(f"unknown training parameters: {sorted(unknown_training)}")
    seed = params["training"].setdefault("seed", 42)
    if type(seed) is not int:
        raise TypeError("training seed must be an integer")
    return params


def _package_version(*distributions: str) -> str:
    for distribution in distributions:
        try:
            return importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unknown"


class _BaseAdapter:
    supported_batches = None

    def __init__(self, spec: dict[str, Any], attempt: Path, kind: str) -> None:
        self.kind = kind
        self.attempt = Path(attempt)
        self.params = _params(spec)
        self._model: Any = None

    def _common_metadata(self) -> dict[str, Any]:
        return {
            "adapter": self.kind,
            "labels": list(LABELS),
            "resolved_params": _json_value(self.params),
            "python_version": platform.python_version(),
            "supported_batches": None,
        }

    def _check_ready(self) -> Any:
        if self._model is None:
            raise RuntimeError(f"{self.kind} adapter is not set up or has been closed")
        return self._model

    def close(self) -> None:
        self._model = None


class _SklearnAdapter(_BaseAdapter):
    def setup(self, train, validation) -> dict[str, Any]:
        try:
            train_rows = _read_split(train, "train")
            validation_rows = _read_split(validation, "validation")
            import sklearn
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            from sklearn.svm import LinearSVC

            vectorizer_params = dict(_TFIDF_DEFAULTS)
            vectorizer_params.update(self.params["preprocessing"])
            if "ngram_range" in vectorizer_params:
                vectorizer_params["ngram_range"] = tuple(
                    vectorizer_params["ngram_range"]
                )
            estimator_params = dict(_SVM_DEFAULTS if self.kind == "svm" else _NB_DEFAULTS)
            if self.kind == "svm":
                estimator_params["random_state"] = self.params["training"]["seed"]
            estimator_params.update(self.params["model"])
            estimator_params = _json_value(estimator_params)
            if self.kind == "svm":
                estimator = LinearSVC(**estimator_params)
            else:
                estimator = MultinomialNB(**estimator_params)
            # Keep the exact resolved configuration in runtime metadata.
            self.params["preprocessing"] = deepcopy(vectorizer_params)
            self.params["model"] = deepcopy(estimator_params)
            model = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(**vectorizer_params)),
                    ("classifier", estimator),
                ]
            )
            model.fit(
                [row[1] for row in train_rows],
                [row[2] for row in train_rows],
            )
            self._model = model
            result = self._common_metadata()
            result.update(_split_metadata(train_rows, validation_rows))
            result.update(
                {
                    "library_versions": {"scikit-learn": sklearn.__version__},
                    "features": {
                        "kind": "TF-IDF",
                        "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
                    },
                    "training": "fit on train split only; no validation-based tuning",
                }
            )
            return _json_value(result)
        except ImportError as exc:
            self.close()
            raise Unavailable(
                f"{self.kind} requires scikit-learn (tested with scikit-learn==1.9.1): {exc}"
            ) from exc
        except MemoryError as exc:
            self.close()
            raise Unavailable(f"{self.kind} ran out of memory during setup") from exc
        except Exception:
            self.close()
            raise

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        model = self._check_ready()
        if not texts:
            return []
        try:
            if self.kind == "svm":
                values = model.decision_function(texts)
                raw_key = "decision_scores"
            else:
                values = model.predict_proba(texts)
                raw_key = "probabilities"
            classes = [LABELS[int(value)] for value in model.classes_]
            predictions = []
            for row in values:
                scores = [float(value) for value in row]
                best = max(range(len(scores)), key=scores.__getitem__)
                predictions.append(
                    {
                        "label": classes[best],
                        "raw": {
                            raw_key: {
                                label: score for label, score in zip(classes, scores)
                            }
                        },
                    }
                )
            return predictions
        except MemoryError as exc:
            raise Unavailable(f"{self.kind} ran out of memory during prediction") from exc


class _FastTextAdapter(_BaseAdapter):
    def _preprocess(self, text: str) -> str:
        options = self.params["preprocessing"]
        if options.get("lowercase", True):
            text = text.lower()
        strip_accents = options.get("strip_accents")
        if strip_accents == "unicode":
            text = "".join(
                char
                for char in unicodedata.normalize("NFKD", text)
                if not unicodedata.combining(char)
            )
        # fastText treats a physical newline as an example boundary.
        return " ".join(text.split())

    def setup(self, train, validation) -> dict[str, Any]:
        training_path: Path | None = None
        try:
            train_rows = _read_split(train, "train")
            validation_rows = _read_split(validation, "validation")
            preprocessing = self.params["preprocessing"]
            unknown_preprocessing = set(preprocessing) - {"lowercase", "strip_accents"}
            if unknown_preprocessing:
                raise ValueError(
                    "unsupported fastText preprocessing parameters: "
                    f"{sorted(unknown_preprocessing)}"
                )
            if preprocessing.get("strip_accents") not in (None, "unicode"):
                raise ValueError("fastText strip_accents must be null or 'unicode'")
            model_params = dict(_FASTTEXT_DEFAULTS)
            model_params["seed"] = self.params["training"]["seed"]
            unknown_model = set(self.params["model"]) - _FASTTEXT_MODEL_OPTIONS
            if unknown_model:
                raise ValueError(
                    "unsupported fastText model parameters (pretrained vectors, "
                    f"autotuning, and label overrides are disabled): {sorted(unknown_model)}"
                )
            model_params.update(self.params["model"])
            self.params["model"] = deepcopy(model_params)
            self.params["preprocessing"].setdefault("lowercase", True)
            self.params["preprocessing"].setdefault("strip_accents", None)

            try:
                import fasttext
            except ImportError as exc:
                raise Unavailable(
                    "fasttext requires fasttext==0.9.3 on CPython 3.13 "
                    "(C++11 source build), or fasttext-wheel==0.9.2 on CPython 3.12"
                ) from exc

            self.attempt.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="fasttext-train-",
                suffix=".txt",
                dir=self.attempt,
                delete=False,
            ) as handle:
                training_path = Path(handle.name)
                for _, text, gold in train_rows:
                    handle.write(
                        f"{FASTTEXT_LABEL_PREFIX}{LABELS[gold]} "
                        f"{self._preprocess(text)}\n"
                    )
            self._model = fasttext.train_supervised(
                input=str(training_path),
                label=FASTTEXT_LABEL_PREFIX,
                **model_params,
            )
            result = self._common_metadata()
            result.update(_split_metadata(train_rows, validation_rows))
            result.update(
                {
                    "library_versions": {
                        "fasttext": _package_version("fasttext", "fasttext-wheel")
                    },
                    "features": {
                        "kind": "fastText supervised bags of words/n-grams",
                        "pretrained_vectors": False,
                    },
                    "training": (
                        "from-scratch fastText train_supervised on train split only; "
                        "validation not used for autotuning"
                    ),
                    "deterministic_configuration": model_params["thread"] == 1,
                }
            )
            return _json_value(result)
        except Unavailable:
            self.close()
            raise
        except MemoryError as exc:
            self.close()
            raise Unavailable("fasttext ran out of memory during setup") from exc
        except Exception:
            self.close()
            raise
        finally:
            if training_path is not None:
                training_path.unlink(missing_ok=True)

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        model = self._check_ready()
        if not texts:
            return []
        try:
            labels, probabilities = model.predict(
                [self._preprocess(text) for text in texts], k=1
            )
            predictions = []
            for raw_labels, raw_probabilities in zip(labels, probabilities):
                raw_label = str(raw_labels[0]) if raw_labels else ""
                label = (
                    raw_label[len(FASTTEXT_LABEL_PREFIX) :]
                    if raw_label.startswith(FASTTEXT_LABEL_PREFIX)
                    else raw_label
                )
                predictions.append(
                    {
                        "label": label if label in LABELS else None,
                        "raw": {
                            "fasttext_label": raw_label,
                            "probability": (
                                float(raw_probabilities[0]) if raw_probabilities else None
                            ),
                        },
                    }
                )
            if len(predictions) != len(texts):
                raise RuntimeError("fastText returned the wrong number of predictions")
            return predictions
        except MemoryError as exc:
            raise Unavailable("fasttext ran out of memory during prediction") from exc


def create(spec: dict, attempt: Path):
    """Create an uninitialised classical adapter from a runner specification."""

    if not isinstance(spec, dict):
        raise TypeError("spec must be a dict")
    kind = spec.get("adapter")
    if kind not in ("svm", "nb", "fasttext"):
        raise ValueError(f"unknown classical adapter: {kind!r}")
    cls = _FastTextAdapter if kind == "fasttext" else _SklearnAdapter
    return cls(spec, Path(attempt), kind)
