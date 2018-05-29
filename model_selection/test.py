from model_selection._split import BootStrapping
import numpy as np
import unittest


class myTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testBootStrapNoWeights(self):
        X = np.arange(100)
        get, not_get = BootStrapping.split(X)
        self.assertAlmostEqual(len(not_get)/100, 1 / np.e, delta=0.1)

    def testBootStrapWeights1(self):
        X = np.array([1, 0])
        sample_weights = np.array([80, 20])
        get, _ = BootStrapping.split(X, sample_weights, iter_num=100)
        p, q = np.unique(get, return_counts=True)
        for e in zip(p, q):
            self.assertAlmostEqual(sample_weights[e[0]] / sum(sample_weights), e[1] / 100, delta=0.1)

    def testBootStrapWeights2(self):
        X = np.arange(10)
        sample_weights = np.random.random(10)
        get, _ = BootStrapping.split(X, sample_weights, iter_num=100)
        p, q = np.unique(get, return_counts=True)
        for e in zip(p, q):
            self.assertAlmostEqual(sample_weights[e[0]] / sum(sample_weights), e[1] / 100, delta=0.1)


if __name__ == '__main__':
    unittest.main()
