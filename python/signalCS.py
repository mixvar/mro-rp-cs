# Compressed sensing
# - rekonstrukcja
# - procesowanie sygnałów, obrazów
# - przekracza granicę częstotliwości Nyquista
# - poszukiwanie odpowiedniej bazy wektorów
# - algebra liniowa


# Imports
from sklearn.linear_model import Lasso
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy as np


# Let's create a sine function with frequency of 50 Hz and keep the following signal for an eighth of a second.
# At a sampling rate of 400 Hz, it comes to 500 samples. As per the example, we take 100 random samples of this signal.


# Initializing constants and signals
N = 500
FS = 4e3 # Signal sample rate
M = 100  # Reconstruction points
f1 = 50  # Frequency
duration = 1. / 8 # Signal duration

# Generate signal
t = np.linspace(0, duration, duration * FS)
sineF = np.sin(2 * np.pi * f1 * t)
f = np.reshape(sineF, (len(sineF), 1))

# Displaying the test signal
plt.plot(t, f)
plt.title('Original Signal')
plt.show()


# Make FFT for this signal.


from scipy.fftpack import fft

yf = fft(sineF)
Nn = len(yf)/2 + 1

dt = t[1] - t[0]
fa = 1.0/dt # scan frequency

print('dt=%.5fs (Sample Time)' % dt)
print('fa=%.2fHz (Frequency)' % fa)

X = np.linspace(0, fa/2, Nn, endpoint=True)
plt.xlabel('Frequency ($Hz$)')
plt.xlim(xmin=0, xmax=1500)
plt.plot(X, np.abs(yf[:int(Nn)]))


# Randomly sampling the test signal
k = np.random.randint(0, N, (M,))
k = np.sort(k)  # making sure the random samples are monotonic
b = f[k]
plt.plot(t, f, 'b', t[k], b, 'r.')
plt.title('Original Signal with Random Samples')
plt.show()


# Since this is a simple, almost stationary signal, a simple basis like discrete cosines should suffice
# to bring out the sparsity.


D = np.fft.fft(np.eye(N))
A = D[k, :]


# Here $A$ is a matrix which contains a subset of 500 discrete cosine bases, and we need to solve Ax=b for x.
# It is a nonlinear optimization problem and there are many solutions,
# but it turns out that the one that minimizes the L_1 norm of the solution gives the best estimate of the original signal.
# Since this is an optimization problem, it can be solved with many of the methods in scipy.optimize,
# say by taking the least squares solution of the equation (or the L_2 norm) as the first guess and minimizing iteratively. But I took the easier approach and used the Lasso estimator in the sklearn package, which is essentially a linear estimator that penalizes (regularizes) its weights in the $L_1$ sense. (A really cool demonstration of compressed sensing for images using Lasso is [here](http://scikit-learn.org/0.14/auto_examples/applications/plot_tomography_l1_reconstruction.html)).


lasso = Lasso(alpha=0.001)
lasso.fit(A, b.reshape((M,)))


# Plotting the reconstructed coefficients and the signal
X2 = np.linspace(0, fa, N, endpoint=True)
plt.plot(X2, lasso.coef_)
plt.xlim([0, 2000])
plt.title('FFT of the Reconstructed Signal')
recons = np.fft.ifft(lasso.coef_.reshape((N, 1)), axis=0)
plt.figure()
plt.plot(t, recons)
plt.title('Reconstucted Signal')
plt.show()


# As can be seen through the plots, most of the coefficients of the lasso estimator as zeros.
# It is the discrete cosine transform of these coefficients that is the reconstructed signal.
# Since the coefficients are sparse, they can be compressed into a scipy.sparse matrix.

recons_sparse = coo_matrix(lasso.coef_)
sparsity = 1 - float(recons_sparse.getnnz()) / len(lasso.coef_)
print(sparsity)


