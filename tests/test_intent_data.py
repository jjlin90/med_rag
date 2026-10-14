"""Prevent exact question leakage and contradictory intent supervision."""
import unittest

from scripts.build_intent_data import split_samples


class IntentDataTests(unittest.TestCase):
    def test_duplicate_questions_cannot_cross_train_and_validation(self):
        samples = [{'text':f'question {label}-{i}', 'label':label}
                   for label in (0, 1) for i in range(20)]
        samples += [dict(samples[0])] * 20
        samples += [{'text':'  question 0-0  ', 'label':0}]
        train, val = split_samples(samples)
        self.assertFalse({r['text'] for r in train} & {r['text'] for r in val})
        self.assertEqual(len(train) + len(val), 40)
        self.assertEqual(sum(r['label'] == 0 for r in train), 18)
        self.assertEqual(sum(r['label'] == 1 for r in val), 2)
        self.assertEqual((train, val), split_samples(samples))

    def test_conflicting_labels_and_invalid_samples_stop_publication(self):
        for samples in ([{'text':'same','label':0}, {'text':'same','label':1}],
                        [{'text':'blank','label':True}],
                        [{'text':'   ','label':0}],
                        [{'text':'one','label':0}, {'text':'two','label':1}]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                split_samples(samples)
