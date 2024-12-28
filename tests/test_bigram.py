import unittest
from src.models.bigram import BigramLanguageModel


class TestBigramModel(unittest.TestCase):

    def setUp(self):
        self.model = BigramLanguageModel()
        self.model.train("the quick brown fox jumps over the lazy dog")

    def test_bigram_probabilities(self):
        prob = self.model.get_bigram_probability("the", "quick")
        self.assertGreater(prob, 0)

    def test_prediction(self):
        prediction = self.model.predict("the")
        self.assertIn(prediction, ["quick", "lazy", "brown"])

    def test_training(self):
        self.model.train("another sentence for training")
        prob = self.model.get_bigram_probability("another", "sentence")
        self.assertGreater(prob, 0)


if __name__ == '__main__':
    unittest.main()
