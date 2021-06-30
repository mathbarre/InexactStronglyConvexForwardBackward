# Inexact Forward-Backward Algorithms for Strongly Convex Objectives

Code associated with the work ['reference'](https://arxiv.org/abs/2106.15536).

> [1] M. Barré, A. B. Taylor and F. Bach, "A note on approximate accelerated forward-backward methods with absolute and relative errors, and possibly strongly convex objectives" arXiv:2106.15536, 2021.

## Authors

- [**Mathieu Barré**](https://mathbarre.github.io/)
- [**Adrien Taylor**](https://www.di.ens.fr/~ataylor/)
- [**Francis Bach**](https://www.di.ens.fr/~fbach/)

## Organization

- The proofs folder contains [mathematica](https://www.wolfram.com/mathematica/) notebooks to help verify the proofs of Theorem 3.1 and 5.1 from the paper.
- The ISCFB folder contains the python code to reproduce experiments appearing in the paper.

## Running the experiments

Before using the python code you should run in the current folder StronglyConvexForwardBackward

```console
pip install -e .
```

Then, to test the method on deblurring with TV regularization you can execute

```console
python ./ISCFB/examples/example_TV.py
```
