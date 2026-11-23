"""Focused tests for the from-scratch classical adapters."""

from __future__ import annotations

import json
from pathlib import Path
from importlib.util import find_spec
import sys
import tempfile
import unittest
from unittest import mock

from adapters.base import Unavailable
from adapters.classical import create


TRAIN = [
    ("n1", "awful terrible bad", 0),
    ("n2", "horrible awful experience", 0),
    ("n3", "bad terrible outcome", 0),
    ("u1", "ordinary neutral statement", 1),
    ("u2", "plain ordinary update", 1),
    ("u3", "neutral factual statement", 1),
    ("p1", "excellent wonderful good", 2),
    ("p2", "happy excellent experience", 2),
    ("p3", "good wonderful outcome", 2),
]
VALIDATION = [("v1", "awful bad", 0), ("v2", "excellent good", 2)]


class ClassicalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.attempt = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_adapter(self, kind="svm", **sections):
        params = {"preprocessing": {"min_df": 1}} if kind != "fasttext" else {}
        params.update(sections)
        return create({"adapter": kind, "params": params}, self.attempt)

    def test_create_is_lazy_and_batches_are_unrestricted(self):
        for kind in ("svm", "nb", "fasttext"):
            with self.subTest(kind=kind):
                adapter = self.make_adapter(kind)
                self.assertIsNone(adapter.supported_batches)
                self.assertIsNone(adapter._model)
                with self.assertRaises(RuntimeError):
                    adapter.predict(["not ready"])
                adapter.close()
                adapter.close()

    def test_unknown_adapter_and_parameters_fail_early(self):
        with self.assertRaises(ValueError):
            create({"adapter": "unknown"}, self.attempt)
        with self.assertRaises(ValueError):
            create(
                {"adapter": "svm", "params": {"unexpected": {}}}, self.attempt
            )

    def test_svm_trains_predicts_and_returns_json_metadata(self):
        adapter = self.make_adapter(
            "svm",
            model={"C": 0.8, "max_iter": 1000},
            training={"seed": 7},
        )
        metadata = adapter.setup(TRAIN, VALIDATION)
        self.assertEqual(metadata["train_rows"], 9)
        self.assertEqual(metadata["validation_rows"], 2)
        self.assertEqual(
            metadata["train_class_counts"],
            {"negative": 3, "neutral": 3, "positive": 3},
        )
        self.assertEqual(metadata["resolved_params"]["model"]["random_state"], 7)
        self.assertEqual(metadata["resolved_params"]["model"]["C"], 0.8)
        self.assertEqual(metadata["resolved_params"]["preprocessing"]["min_df"], 1)
        self.assertGreater(metadata["features"]["vocabulary_size"], 0)
        self.assertFalse(metadata["validation_used_for_training_or_tuning"])

        texts = ["awful terrible", "ordinary neutral", "excellent wonderful"]
        predictions = adapter.predict(texts)
        self.assertEqual(
            [item["label"] for item in predictions],
            ["negative", "neutral", "positive"],
        )
        for item in predictions:
            self.assertEqual(set(item["raw"]["decision_scores"]), set(metadata["labels"]))
            self.assertTrue(
                all(
                    isinstance(value, float)
                    for value in item["raw"]["decision_scores"].values()
                )
            )
        json.dumps(metadata, allow_nan=False)
        json.dumps(predictions, allow_nan=False)
        self.assertEqual(adapter.predict(texts), predictions, "SVM must be repeatable")
        self.assertEqual(adapter.predict([]), [])
        adapter.close()

    def test_nb_uses_same_configurable_tfidf_pipeline(self):
        adapter = self.make_adapter(
            "nb",
            preprocessing={
                "min_df": 1,
                "ngram_range": [1, 1],
                "sublinear_tf": False,
            },
            model={"alpha": 0.5},
        )
        metadata = adapter.setup(TRAIN, [])
        self.assertEqual(
            metadata["resolved_params"]["preprocessing"]["ngram_range"], [1, 1]
        )
        self.assertFalse(metadata["resolved_params"]["preprocessing"]["sublinear_tf"])
        self.assertEqual(metadata["resolved_params"]["model"]["alpha"], 0.5)
        predictions = adapter.predict(
            ["awful terrible", "ordinary neutral", "excellent wonderful"]
        )
        self.assertEqual(
            [item["label"] for item in predictions],
            ["negative", "neutral", "positive"],
        )
        for item in predictions:
            probabilities = item["raw"]["probabilities"]
            self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        json.dumps(predictions, allow_nan=False)
        adapter.close()

    def test_invalid_training_data_is_rejected(self):
        for rows in ([], TRAIN[:3], [("bad", "text", 3)]):
            with self.subTest(rows=rows):
                adapter = self.make_adapter("nb")
                with self.assertRaises(ValueError):
                    adapter.setup(rows, [])
                adapter.close()
        adapter = self.make_adapter("nb")
        with self.assertRaises(TypeError):
            adapter.setup([("bad", None, 0)], [])

    def test_fasttext_from_scratch_training_preprocessing_and_prediction(self):
        fake = _FakeFastText()
        with mock.patch.dict(sys.modules, {"fasttext": fake}):
            adapter = self.make_adapter(
                "fasttext",
                preprocessing={"lowercase": True, "strip_accents": "unicode"},
                model={"epoch": 3, "thread": 1},
                training={"seed": 19},
            )
            train = TRAIN + [("extra", "Café\nPOSITIVE\tnews", 2)]
            metadata = adapter.setup(train, VALIDATION)
            self.assertIsNotNone(adapter._model)
            self.assertEqual(fake.kwargs["epoch"], 3)
            self.assertEqual(fake.kwargs["seed"], 19)
            self.assertEqual(fake.kwargs["thread"], 1)
            self.assertEqual(fake.kwargs["label"], "__label__")
            self.assertNotIn("pretrainedVectors", fake.kwargs)
            self.assertIn("__label__negative awful terrible bad\n", fake.training_data)
            self.assertIn("__label__positive cafe positive news\n", fake.training_data)
            self.assertFalse(Path(fake.kwargs["input"]).exists())
            self.assertFalse(metadata["features"]["pretrained_vectors"])
            self.assertTrue(metadata["deterministic_configuration"])

            first = adapter.predict(["POSITIVE excellent", "awful BAD", "neutral NEWS"])
            second = adapter.predict(["POSITIVE excellent", "awful BAD", "neutral NEWS"])
            self.assertEqual(first, second, "predictions must not be cached")
            self.assertEqual(fake.model.calls, 2)
            self.assertEqual(
                [item["label"] for item in first], ["positive", "negative", "neutral"]
            )
            self.assertIsInstance(first[0]["raw"]["probability"], float)
            json.dumps(first, allow_nan=False)
            adapter.close()
            with self.assertRaises(RuntimeError):
                adapter.predict(["closed"])

    def test_fasttext_rejects_pretrained_vectors_and_autotuning(self):
        for option in ("pretrainedVectors", "autotuneValidationFile", "label"):
            with self.subTest(option=option):
                adapter = self.make_adapter("fasttext", model={option: "forbidden"})
                with self.assertRaises(ValueError):
                    adapter.setup(TRAIN, VALIDATION)

    def test_fasttext_memory_error_is_an_unavailable_skip(self):
        fake = _FakeFastText(fail_with_memory_error=True)
        with mock.patch.dict(sys.modules, {"fasttext": fake}):
            adapter = self.make_adapter("fasttext")
            with self.assertRaises(Unavailable):
                adapter.setup(TRAIN, VALIDATION)
            self.assertIsNone(adapter._model)

    @unittest.skipUnless(find_spec("fasttext"), "fastText is an optional test dependency")
    def test_fasttext_real_native_smoke(self):
        adapter = self.make_adapter(
            "fasttext", model={"epoch": 1, "dim": 10, "bucket": 1000, "thread": 1}
        )
        metadata = adapter.setup(TRAIN, VALIDATION)
        predictions = adapter.predict(["awful bad", "ordinary neutral", "excellent good"])
        self.assertEqual(len(predictions), 3)
        self.assertTrue(
            all(item["label"] in ("negative", "neutral", "positive") for item in predictions)
        )
        self.assertNotEqual(metadata["library_versions"]["fasttext"], "unknown")
        json.dumps(predictions, allow_nan=False)
        adapter.close()


class _FakeFastTextModel:
    def __init__(self):
        self.calls = 0

    def predict(self, texts, k=1):
        self.calls += 1
        labels = []
        probabilities = []
        for text in texts:
            if "positive" in text or "excellent" in text:
                label = "positive"
            elif "bad" in text or "awful" in text:
                label = "negative"
            else:
                label = "neutral"
            labels.append((f"__label__{label}",))
            probabilities.append((0.75,))
        return labels, probabilities


class _FakeFastText:
    """Module-like fastText double; training input is captured before cleanup."""

    def __init__(self, fail_with_memory_error=False):
        self.fail_with_memory_error = fail_with_memory_error
        self.model = _FakeFastTextModel()
        self.kwargs = None
        self.training_data = None

    def train_supervised(self, **kwargs):
        self.kwargs = kwargs
        self.training_data = Path(kwargs["input"]).read_text(encoding="utf-8")
        if self.fail_with_memory_error:
            raise MemoryError("synthetic allocation failure")
        return self.model


if __name__ == "__main__":
    unittest.main()
