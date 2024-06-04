from solvers_plotters import *


compound_block = comp_sym_corr(0.8, 3)
block_diag_matrix12 = np.block([[compound_block, np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))],
                                [np.zeros((3,3)), compound_block, np.zeros((3,3)), np.zeros((3,3))],
                                [np.zeros((3,3)), np.zeros((3,3)), compound_block, np.zeros((3,3))],
                                [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), compound_block]])


# Simulation study comparing asymptotic performance of Lasso, Fused Lasso and SLOPE:
# We consider three different signal vectors b_0: a) [0,0,1,0], b) [1,1,1,1], c) [1,0,1,0]

rho = 0.8
plot_performance(b_0=np.array([0, 0, 1, 0]),
                 C=np.array([[1, 0, rho, 0], [0, 1, 0, rho], [rho, 0, 1, 0], [0, rho, 0, 1]]),
                 lambdas= np.array([1.6, 1.2, 0.8, 0.4]), #np.array([4, 0, 0, 0]),
                 x=np.linspace(0,1,20),
                 n=1000, # increase for more accurate results, in the paper n=15000 was used
                 Cov=0.4**2*np.array([[1, 0, rho, 0], [0, 1, 0, rho], [rho, 0, 1, 0], [0, rho, 0, 1]]),
                 flasso=True,
                 A_flasso=Acustom(a=np.ones(4), b=1 * np.ones(3)),
                 reducedOLS=False, # set True for reduced OLS
                 sigma=0.4,
                 smooth=True,
                 legend=True)


# Simulation study comparing asymptotic performance of Lasso, Fused Lasso, Concavified Fused Lasso and SLOPE
# Specifying the ``Concavified Fused Lasso'' penalty ||Ab||_1, where A is a (2p-1) x p matrix. (special case of Generalized Lasso)

curvature = 0.04 # curvature/concavity parameter
cluster_scaling = 0.8 # clustering parameter
A12concave = Aconcave(12, curvature, cluster_scaling) # Concavified Fused Lasso penalty matrix
A12flasso = Acustom(a=np.ones(12), b=np.ones(11) * sum(A12concave[i][i] for i in range(12)) * (1 / 11)) # Fused Lasso penalty matrix
# A12flasso is normalized so that total clustering is the same as in A12concave
print('Aconcave:\n', Aconcave(12, 0.04, 0.8))
print('A12flasso:\n', np.round(A12flasso, 3))
#print('lin_lambdas(12):', lin_lambdas(12))


plot_performance(b_0=np.array([0, 0, 0, 1, 1, 1, 3, 3, 3, 2, 2, 2]),  # np.array([1, 1, 1, 0, 0, 0, 3, 3, 3, 2, 2, 2])
                 C=block_diag_matrix12,
                 lambdas=lin_lambdas(12),
                 x=np.linspace(0, 2, 24),
                 n=100, # increase for more accurate results, in the paper n=15000 was used
                 Cov=0.2**2*block_diag_matrix12,  #block_diag_matrix12
                 flasso=True,
                 A_flasso=A12flasso,
                 glasso=True,
                 A_glasso=A12concave,
                 reducedOLS=False, # set True for reduced OLS
                 sigma=0.2,
                 smooth=True,
                 tol=1e-4
                 )




#Phase Transition in Pattern recovery
alpha1 = 2/3-0.05
alpha2 = 2/3
alpha3 = 2/3+0.05
C1 = np.array([[1, alpha1], [alpha1, 1]])
C2 = np.array([[1, alpha2], [alpha2, 1]])
C3 = np.array([[1, alpha3], [alpha3, 1]])
sigma=0.2
# the break points are sparse and the plot is only illustrative, showcasing the phase transition
custom_points = np.array([0, 0.05, 0.18, 0.5, 1.2, 2])


plot_performance_tripple(b_0=np.array([1, 0]),
                         C1=C1,
                         C2=C2,
                         C3=C3,
                         lambdas=np.array([3, 2]),
                         x = custom_points,
                         n=500,
                         Cov1=sigma ** 2 * C1,
                         Cov2= 0.2 ** 2 * C2,
                         Cov3=sigma ** 2 * C3)


