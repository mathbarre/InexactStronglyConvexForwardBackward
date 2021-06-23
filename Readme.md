# Inexact Forward-Backward Algorithms for Strongly Convex Objectives

Code associated with the work TODO ['reference'](https://arxiv.org/)

### Authors

- [**Mathieu Barré**](https://mathbarre.github.io/)
- [**Adrien Taylor**](https://www.di.ens.fr/~ataylor/)
- [**Francis Bach**](https://www.di.ens.fr/~fbach/)

### Organization

- The proofs folder contains [mathematica](https://www.wolfram.com/mathematica/) notebooks to help verify the proofs of Theorem 3.1 and 5.1 from the paper.
- The ISCFB folder contains the python code to reproduce experiments appearing in the paper.

### Running the experiments

Before using the python code you should run in the current folder StronglyConvexForwardBackward

```console
pip install -e .
```

Then, to test the method on deblurring with TV regularization you can execute

```console
python ./ISCFB/examples/example_TV.py
```
