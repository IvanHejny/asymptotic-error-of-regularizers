import numpy as np
from solvers_slope import *
#from solvers_glasso import *

b_0_test3 = np.array([0, 2, 0, 2, -2, -2, 1, 1])
y_test3 = np.array([5.0, 60.0, 4.0, 50.0, 10.0, -5.0, 12.0, 17.0])

lambda_test3 = np.array([65.0, 42.0, 40.0, 20.0, 18.0, 15.0, 3.0, 1.0])
lambda_test4 = [35.0, 35.0, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2]
lambda_test5 = [35.0, 35.0, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8]

print('prox_slope_b_0_3:', prox_slope_b_0(b_0_test3, y_test3, lambda_test3))
print('prox_slope_b_0:', prox_slope_b_0(b_0_test3, y_test3, np.flip(lambda_test3)))# flipping lambdas has no effect (sanity check)
print('prox_slope_b_0_4:', prox_slope_b_0(b_0_test3, y_test3, lambda_test4))
print('prox_slope_b_0_5:', prox_slope_b_0(b_0_test3, y_test3, lambda_test5))

C_test3 = np.identity(8)
W_test3 = np.array([5.0, -2.0, 3.0, 3.1, -2.5, -5.2, 0.7, -7.0])
for i in range(20,25):
    print("pdg_slope_b_0_FISTA:", pgd_slope_b_0_FISTA(np.identity(8), W_test3, b_0_test3, lambda_test3, 0.3, i))

# Compare with
#print('b_0:', b_0_test3)
#print("zero-cluster:", prox_slope(y=np.array([5.0, 4.0]), lambdas=np.array([3.0, 1.0])),
#      "one-cluster:", prox_slope_on_b_0_single_cluster(b_0=np.array([1, 1]), y=np.array([12.0, 17.0]),
#                                                       lambdas=np.array(np.array([18.0, 15.0]))),
#      "two-cluster:",
#      prox_slope_on_b_0_single_cluster(b_0=np.array([2, 2, -2, -2]), y=np.array([60.0, 50.0, 10.0, -5.0]),
#                                       lambdas=np.array([65.0, 42.0, 40.0, 20.0])))
