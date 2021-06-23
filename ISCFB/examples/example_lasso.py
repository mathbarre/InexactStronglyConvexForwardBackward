import numpy as np
from libsvmdata import fetch_libsvm
from ISCFB.Inexact_FB import inexact_fb
from ISCFB.utils import ST_vec, strFistaLasso, zero
import matplotlib.pyplot as plt
# data generation

dataset = "madelon"
# dataset = "rcv1.binary"

X, y = fetch_libsvm(dataset, normalize=False)

X = X.toarray()

L = np.linalg.norm(X, ord=2)**2

true_m = 0.00001*L
l_reg = X.shape[0]*0.1

x0 = np.zeros(X.shape[1])


def f(x):
    return 0.5*np.sum((X@x - y)**2)


def fmu(x):
    return 0.5*np.sum((X@x - y)**2)+true_m/2*np.sum(x**2)


def gradf(x):
    return X.T@(X@x-y)


def gradfmu(x):
    return X.T@(X@x-y) + true_m*x


def g(x):
    return np.sum(l_reg*abs(x) + true_m/2*x**2)


def g_nomu(x):
    return np.sum(l_reg*abs(x))


def proxg(y, gradf_y, stepsize, maxiter_inner, sigma, zeta, xi, m):
    res = ST_vec((y-stepsize*gradf_y)/(1+true_m*stepsize),
                 stepsize/(1+true_m*stepsize)*l_reg)
    v = (y-stepsize*gradf_y-res)/stepsize
    return (res, v, 1, 0)


def proxg_nomu(y, gradf_y, stepsize, maxiter_inner, sigma, zeta, xi, m):
    res = ST_vec(y-stepsize*gradf_y, stepsize*l_reg)
    v = (y-stepsize*gradf_y-res)/stepsize
    return (res, v, 1, 0)


sigma = 0
zeta = 0
xi = zero

(x_no_mu, info_no_mu) = inexact_fb(x0, f, g, gradf, proxg, L, 0, sigma, zeta,
                                   xi, maxiter_tot=20000, maxiter_inner=1000,
                                   backtrack=0, verbose=1, freq=200)


(x, info) = inexact_fb(x0, f, g, gradf, proxg, L, true_m, sigma, zeta, xi,
                       maxiter_tot=20000, maxiter_inner=1000, backtrack=0,
                       verbose=1, freq=200)

(x_musmooth, info_musmooth) = inexact_fb(x0, fmu, g_nomu, gradfmu, proxg_nomu,
                                         L+true_m, true_m, sigma, zeta, xi,
                                         maxiter_tot=20000, maxiter_inner=1000,
                                         backtrack=0, verbose=1, freq=200)

(xfista, res_fista) = strFistaLasso(X, y, x0, L, true_m, 20000, l_reg)

mf = min(np.min(res_fista), np.min(info_musmooth[:, 0]))

plt.semilogy(np.arange(len(info_no_mu[:, 0])), info_no_mu[:, 0]-mf,
             label="no mu")
plt.semilogy(np.arange(len(info[:, 0])), info[:, 0]-mf, label="known mu")
plt.semilogy(np.arange(len(info_musmooth[:, 0])), info_musmooth[:, 0]-mf,
             label="mu in smooth part")
plt.semilogy(np.arange(len(res_fista)), res_fista-mf, label="fista")
plt.legend()
plt.show()
# %%
