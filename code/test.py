import unittest
from code.solvers_slope import *
from code.solvers_glasso import *

class test_solvers_slope(unittest.TestCase):
    def test_prox_slope_on_b_0_single_cluster(self):

        # First test
        test1 = prox_slope_on_b_0_single_cluster(b_0=np.array([2, 2, -2, -2]),
                                                 y=np.array([60.0, 50.0, 10.0, -5.0]),
                                                 lambdas=np.array([65.0, 42.0, 40.0, 20.0]))
        true1 = np.array([1.5, 1.5, 32.5, 32.5])
        np.testing.assert_array_equal(test1, true1)

        # Second test
        test2 = prox_slope_on_b_0_single_cluster(b_0=np.array([1, 1]),
                                                 y=np.array([12.0, 17.0]),
                                                 lambdas=np.array(np.array([18.0, 15.0])))

        true2 = np.array([-3, -1])
        np.testing.assert_array_equal(test2, true2)

    def test_prox_slope_b_0(self):
        # testing the proximal operator of the directional SLOPE derivative:
        # First test
        test1 = prox_slope_b_0(b_0=np.array([0, 0, 0, 0]),
                               y=np.array([60, 50, 10, -5]),
                               lambdas=np.array([65, 42, 40, 20]))
        true1 = np.array([1.5, 1.5, 0, 0])
        np.testing.assert_array_equal(test1, true1)

        # Second test
        test2 = prox_slope_b_0(b_0=np.array([0, 0, 0, 0]),
                               y=np.array([60, 50, 10, -46]),
                               lambdas=np.array([65, 42, 40, 20]))
        true2 = np.array([3, 3, 0, -3])
        np.testing.assert_array_equal(test1, true1)

        # Third test
        test3 = prox_slope_b_0(b_0 = np.array([0, 2, 0, 2, -2, -2, 1, 1]),
                              y = np.array([5, 60, 4, 50, 10, -5, 12, 17]),
                              lambdas = np.array([65, 42, 40, 20, 18, 15, 3, 1]))
        true3 = np.array([ 2.5,  1.5,  2.5,  1.5, 32.5, 32.5, -3,  -1 ])
        np.testing.assert_array_equal(test2, true2)

if __name__ == '__main__':
    unittest.main()

class test_slovers_glasso(unittest.TestCase):
    def test_soft_thresholding(self):
        test = soft_thresholding(a=np.array([-3, -2, 0, 1.4, 2.7]), kappa=1)
        test = np.round(test, 15)  # rounding to avoid floating point errors
        true = np.array([-2, -1, 0, 0.4, 1.7])
        np.testing.assert_array_equal(test, true)
    def test_admm_glasso(self):
        # First test
        test1 = admm_glasso(C=np.array([[1, 0.8], [0.8, 1]]),
                            A=np.array([[1, -1], [1, 0], [0, 1]]),
                            w=np.array([1, 2.08]),
                            beta0=np.array([0, 0]),
                            lambdas=1.0,
                            iter=100)
        test1 = np.round(test1, 15)  # rounding to avoid floating point errors
        true1 = np.array([0.3, 0.3])

        np.testing.assert_array_equal(test1, true1)

if __name__ == '__main__':
    unittest.main()
