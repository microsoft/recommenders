# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
Module maintaining the IMC problem.
"""

import numpy as np

from scipy.sparse import csr_matrix
from numba import njit, prange
from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Product, SymmetricPositiveDefinite
from pymanopt.autodiff.backends import numpy as backend_decorator
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers.line_search import BackTrackingLineSearcher


class IMCProblem(object):
    """
    Implements the IMC problem.
    """

    def __init__(self, dataPtr, lambda1=1e-2, rank=10):
        """Initialize parameters

        Args:
            dataPtr (DataPtr): An object of which contains X, Z side features and target matrix Y.
            lambda1 (uint): Regularizer.
            rank (uint): rank of the U, B, V parametrization.
        """

        self.dataset = dataPtr
        self.X = self.dataset.get_entity("row")
        self.Z = self.dataset.get_entity("col")
        self.rank = rank
        self._loadTarget()
        self.shape = (self.X.shape[0], self.Z.shape[0])
        self.lambda1 = lambda1
        self.nSamples = self.Y.data.shape[0]

        self.W = None
        self.optima_reached = False
        self.manifold = Product(
            [
                Stiefel(self.X.shape[1], self.rank),
                SymmetricPositiveDefinite(self.rank),
                Stiefel(self.Z.shape[1], self.rank),
            ]
        )

    def _loadTarget(
        self,
    ):
        """Loads target matrix from the dataset pointer."""
        self.Y = self.dataset.get_data()

    @staticmethod
    @njit(nogil=True, parallel=True)
    def _computeLoss_csrmatrix(a, b, cd, indices, indptr, residual_global):
        """computes residual_global = a*b - cd at given indices in csr_matrix format"""
        N = a.shape[0]
        M = a.shape[1]
        for i in prange(N):
            for j in prange(indptr[i], indptr[i + 1]):
                num = 0.0
                for k in range(M):
                    num += a[i, k] * b[k, indices[j]]
                residual_global[j] = num - cd[j]
        return residual_global

    def _cost(self, U, S, VT, residual_global):
        """Compute the cost of GeoIMC optimization problem

        Args:
            params (Iterator): An iterator containing the manifold point at which
            the cost needs to be evaluated.
            residual_global (csr_matrix): Residual matrix.
        """
        regularizer = 0.5 * self.lambda1 * np.sum(S**2)

        IMCProblem._computeLoss_csrmatrix(
            self.X.dot(U.dot(S)),
            VT.T.dot(self.Z.T),
            self.Y.data,
            self.Y.indices,
            self.Y.indptr,
            residual_global,
        )
        cost = 0.5 * np.sum((residual_global) ** 2) / self.nSamples + regularizer

        return cost

    def _egrad(self, U, S, VT, residual_global):
        """Computes the euclidean gradient

        Args:
            params (Iterator): An iterator containing the manifold point at which
            the cost needs to be evaluated.
            residual_global (csr_matrix): Residual matrix.
        """
        residual_global_csr = csr_matrix(
            (residual_global, self.Y.indices, self.Y.indptr),
            shape=self.shape,
        )

        gradU = (
            np.dot(self.X.T, residual_global_csr.dot(self.Z.dot(VT.dot(S.T))))
            / self.nSamples
        )

        gradB = (
            np.dot((self.X.dot(U)).T, residual_global_csr.dot(self.Z.dot(VT)))
            / self.nSamples
            + self.lambda1 * S
        )
        gradB_sym = (gradB + gradB.T) / 2

        gradV = (
            np.dot((self.X.dot(U.dot(S))).T, residual_global_csr.dot(self.Z)).T
            / self.nSamples
        )

        return [gradU, gradB_sym, gradV]

    def solve(self, *args):
        """Main solver of the IMC model

        Args:
            max_opt_time (uint): Maximum time (in secs) for optimization
            max_opt_iter (uint): Maximum iterations for optimization
            verbosity (uint): The level of verbosity for Pymanopt logs
        """
        if self.optima_reached:
            return

        self._optimize(*args)

        self.optima_reached = True
        return

    def _optimize(self, max_opt_time, max_opt_iter, verbosity):
        """Optimize the GeoIMC optimization problem

        Args: The args of `solve`
        """
        residual_global = np.zeros(self.Y.data.shape)

        solver = ConjugateGradient(
            max_time=max_opt_time,
            max_iterations=max_opt_iter,
            line_searcher=BackTrackingLineSearcher(),
            verbosity=verbosity,
        )

        @backend_decorator(self.manifold)
        def _cost(u, s, vt):
            return self._cost(u, s, vt, residual_global)

        @backend_decorator(self.manifold)
        def _egrad(u, s, vt):
            return self._egrad(u, s, vt, residual_global)

        prb = Problem(
            manifold=self.manifold,
            cost=_cost,
            euclidean_gradient=_egrad,
        )
        solution = solver.run(prb, initial_point=self.W)
        self.W = [solution.point[0], solution.point[1], solution.point[2]]

        return solution.cost

    def reset(self):
        """Reset the model."""
        self.optima_reached = False
        self.W = None
        return
