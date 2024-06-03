import unittest
from solvers_slope import *
#from solvers_glasso import *

class test_solvers_slope(unittest.TestCase):
    # First test case
    test1 = prox_slope_on_b_0_single_cluster(b_0=np.array([2, 2, -2, -2]),
                                             y=np.array([60.0, 50.0, 10.0, -5.0]),
                                             lambdas=np.array([65.0, 42.0, 40.0, 20.0]))
    true1 = np.array([1.5, 1.5, 32.5, 32.5])
    np.testing.assert_array_equal(test1, true1)

    # Second test case
    test2 = prox_slope_on_b_0_single_cluster(b_0=np.array([1, 1]),
                                             y=np.array([12.0, 17.0]),
                                             lambdas=np.array(np.array([18.0, 15.0])))

    true2 = np.array([-3, -1])
    np.testing.assert_array_equal(test2, true2)

    def test_prox_slope_on_b_0_single_cluster(self):
        test = prox_slope_on_b_0_single_cluster(b_0 = np.array([2, 2, -2, -2]),
                                   y = np.array([60.0, 50.0, 10.0, -5.0]),
                             lambdas = np.array([65.0, 42.0, 40.0, 20.0]))
        true = np.array([ 1.5,  1.5, 32.5, 32.5])
        np.testing.assert_array_equal(test, true)


if __name__ == '__main__':
    unittest.main()


print(prox_slope_on_b_0_single_cluster(b_0 = np.array([2, 2, -2, -2]),
                                   y = np.array([60.0, 50.0, 10.0, -5.0]),
                             lambdas = np.array([65.0, 42.0, 40.0, 20.0])))

assert all (prox_slope_on_b_0_single_cluster(b_0 = np.array([2, 2, -2, -2]),
                                   y = np.array([60.0, 50.0, 10.0, -5.0]),
                             lambdas = np.array([65.0, 42.0, 40.0, 20.0])) == np.array([ 1.5,  1.5, 32.5, 32.5])), "Should be [ 2.5,  1.5, 32.5, 32.5]"


# Test for proximal operator of the SLOPE directional derivative at b_0

b_0_test3 = np.array([0, 2, 0, 2, -2, -2, 1, 1])
y_test3 = np.array([5.0, 60.0, 4.0, 50.0, 10.0, -5.0, 12.0, 17.0])

lambda_test3 = np.array([65.0, 42.0, 40.0, 20.0, 18.0, 15.0, 3.0, 1.0])
lambda_test4 = [35.0, 35.0, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2]
lambda_test5 = [35.0, 35.0, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8]

print('prox_slope_b_0_3:', prox_slope_b_0(b_0_test3, y_test3, lambda_test3))
assert all(prox_slope_b_0(b_0_test3, y_test3, lambda_test3) == np.array([ 2.5,  1.5,  2.5,  1.5, 32.5, 32.5, -3.,  -1 ])), "Should be [ 2.5,  1.5,  2.5,  1.5, 32.5, 32.5, -3,  -1 ]"

print('prox_slope_b_0:', prox_slope_b_0(b_0_test3, y_test3, np.flip(lambda_test3)))# flipping lambdas has no effect (sanity check)
print('prox_slope_b_0_4:', prox_slope_b_0(b_0_test3, y_test3, lambda_test4))
print('prox_slope_b_0_5:', prox_slope_b_0(b_0_test3, y_test3, lambda_test5))

C_test3 = np.identity(8)
W_test3 = np.array([5.0, -2.0, 3.0, 3.1, -2.5, -5.2, 0.7, -7.0])


# Compare with
#print('b_0:', b_0_test3)
#print("zero-cluster:", prox_slope(y=np.array([5.0, 4.0]), lambdas=np.array([3.0, 1.0])),
#      "one-cluster:", prox_slope_on_b_0_single_cluster(b_0=np.array([1, 1]), y=np.array([12.0, 17.0]),
#                                                       lambdas=np.array(np.array([18.0, 15.0]))),
#      "two-cluster:",
#      prox_slope_on_b_0_single_cluster(b_0=np.array([2, 2, -2, -2]), y=np.array([60.0, 50.0, 10.0, -5.0]),
#                                       lambdas=np.array([65.0, 42.0, 40.0, 20.0])))
