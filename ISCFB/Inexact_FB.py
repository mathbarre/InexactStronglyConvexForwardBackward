import numpy as np
from math import sqrt


def inexact_fb(X0, f, g, gradf, proxg, L, m, sigma, zeta, xi,
               maxiter_tot=1000, maxiter_inner=1000, backtrack=1, verbose=1,
               freq=10):

    X = X0.copy()
    Z = X0.copy()
    fx = f(X)
    gx = g(X)
    infos = np.array([fx+gx, 0, 0])
    Ak = 0
    k = 1
    while True:
        X_old = X.copy()
        stepsize = (1-sigma**2)/L
        eta = (1-zeta**2)*stepsize
        ak = (eta + 2*Ak*m*eta + sqrt(4*eta*Ak*(1+m*eta)*(1+Ak*m)
                                      + eta**2))/2

        Y = X_old + ak*(1+Ak*m)/(ak + Ak + Ak*(2*ak+Ak)*m)*(Z-X_old)

        gradf_y = gradf(Y)

        (X, V, proxiter, proxgap) = proxg(Y, gradf_y,
                                          stepsize, maxiter_inner,
                                          sigma, zeta, xi(k), m)

        fx = f(X)
        gx = g(X)
        infos = np.vstack([infos, np.array([fx+gx, proxiter, proxgap])])
        if verbose and (k % freq == 0):
            print("k = %i, proxIters = %i, proxGap = %e, f = %e, L = %e" %
                  (k, proxiter, proxgap, fx+gx, L))

        if backtrack:
            fy = f(Y)
            gradf_x = gradf(X)
            while (fy < fx + gradf_x.flatten().dot((Y-X).flatten())
                   + 1/2/L*np.sum(((gradf_x-gradf_y).flatten())**2)):
                if verbose:
                    print("Lipschitz estimate too small, increasing...")
                L *= 2
                stepsize = (1-sigma**2)/L
                eta = (1-zeta**2)*stepsize
                ak = (eta + 2*Ak*m*eta + sqrt(4*eta*Ak*(1+m*eta)*(1+Ak*m)
                                              + eta**2))/2

                Y = X_old + ak*(1+Ak*m)/(ak + Ak + Ak*(2*ak+Ak)*m)*(Z-X_old)

                gradf_y = gradf(Y)

                (X, V, proxiter, proxgap) = proxg(Y, gradf_y,
                                                  stepsize, maxiter_inner,
                                                  sigma, zeta, xi(k), m)

                fx = f(X)
                gx = g(X)
                fy = f(Y)
                gradf_x = gradf(X)
                infos = np.vstack([infos,
                                   np.array([fx+gx, proxiter, proxgap])])
                if verbose and (k % freq == 0):
                    print("k = %i, proxIters = %i, "
                          "proxGap = %e, f = %e, L = %e"
                          % (k, proxiter, proxgap, fx+gx, L))
                if np.sum(infos[:, 1]) >= maxiter_tot:
                    return (X, infos)
        # infos = np.vstack([infos,
        #                   np.array([fx+gx, proxiter, proxgap])])
        Z[:] -= ak/(1+m*(ak+Ak))*(V+gradf_y-m*(X-Z))
        Ak = ak+Ak

        k += 1

        if np.sum(infos[:, 1]) >= maxiter_tot:
            return (X, infos)

        if backtrack:
            L /= 1.1
