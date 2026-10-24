import unittest
from types import SimpleNamespace

from utils.vision_engine import process_frame


class FakeResult:
    def __init__(self, class_ids):
        self.boxes = [
            SimpleNamespace(cls=[class_id]) for class_id in class_ids
        ]

    def plot(self):
        return [int(box.cls[0]) for box in self.boxes]


class FakeModel:
    def __init__(self, names, detected_class_ids):
        self.names = names
        self.detected_class_ids = detected_class_ids
        self.calls = []

    def __call__(self, frame, *, classes, verbose):
        self.calls.append({"frame": frame, "classes": classes, "verbose": verbose})
        return [FakeResult(self.detected_class_ids)]


class ProcessFrameTests(unittest.TestCase):
    def test_excludes_person_and_person_aliases_from_inference_and_results(self):
        model = FakeModel(
            {0: "bottle", 1: "person", 2: "human", 3: "umbrella"},
            [0, 1, 2, 3],
        )
        database = {
            "plastic bottle": {},
            "person": {},
            "human": {},
            "people": {},
        }

        annotated, detected = process_frame("frame", model, database)

        self.assertEqual(annotated, [0])
        self.assertEqual(detected, ["plastic bottle"])
        self.assertEqual(model.calls[0]["classes"], [0])

    def test_matches_aliases_with_case_and_whitespace_normalization(self):
        model = FakeModel(
            {0: " Bottle ", 1: "MOUSE", 2: "cell phone"},
            [0, 1, 2],
        )
        database = {
            "Plastic   Bottle": {},
            "Computer Mouse": {},
            "phone": {},
        }

        _, detected = process_frame("frame", model, database)

        self.assertEqual(
            detected,
            ["Plastic   Bottle", "Computer Mouse", "phone"],
        )
        self.assertEqual(model.calls[0]["classes"], [0, 1, 2])

    def test_does_not_report_classes_without_database_mapping(self):
        model = FakeModel(
            {0: "bottle", 1: "umbrella", 2: "person"},
            [0, 1, 2],
        )

        _, detected = process_frame(
            "frame",
            model,
            {"plastic bottle": {}, "person": {}},
        )

        self.assertEqual(detected, ["plastic bottle"])
        self.assertEqual(model.calls[0]["classes"], [0])

    def test_raises_clear_error_when_no_non_person_class_is_supported(self):
        model = FakeModel(
            {0: "person", 1: "human", 2: "umbrella"},
            [],
        )

        with self.assertRaisesRegex(ValueError, "Supported labels: umbrella"):
            process_frame("frame", model, {"person": {}, "human": {}})
        self.assertEqual(model.calls, [])

    def test_explains_when_model_exposes_only_people(self):
        model = FakeModel({0: "person", 1: "human"}, [])

        with self.assertRaisesRegex(ValueError, "no non-person classes"):
            process_frame("frame", model, {"person": {}, "people": {}})
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
