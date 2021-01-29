# Created by Amir.H Ebrahimi at 1/26/21
# Copyright © 2021 Carbon, Inc. All rights reserved.
# Implementation of SVMs multi-class loss feedback based discriminative dictionary learning for image classification (SMLFDL) algorithm

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import DictionaryLearning, PCA

np.random.seed(seed=13)
regularization = namedtuple("REGULARIZATION", "e e1 g")
EPS = 1e-2


class SMLFDL:
    """
    X   mxn :m=features :n=data count             - training data
    y   1xn :n=count                              - training label class
    D   mxK :m=features :K=atoms count            - Dictionary K is k for each class
        - K % |classes| = 0                       - Dictionary components= K // |classes|
        - K / |classes| <= min(Xc.shape)          - [PCA_BOUND] for all subset Xc[y == c]
        - m < K << n                              - from two above?
    Z   Kxn :K=atoms count :n=data count          - Used in loss function?
    P   KxK :K=atoms count
    Q   KxK :K=atoms count
    Uz  1*c :c=number of more probable classes
        - c < |classes|

    reg = regularization(λ, λ1, ɣ)
    """

    def __init__(self, K, reg, max_iteration=1):
        self.K = K  # number of atoms
        self.D = None  # dictionary matrix
        self.Z = None  # coefficients matrix
        self.reg = reg  # regularization {lambda lambda1 gamma}
        self.classes_ = None  # unique y classes
        self.max_iteration = max_iteration  # maximum iteration count
        self.epoch = None
        # Learning algorithms
        self.dictionary_learning = None
        self._update_dictionary_learning()
        # https://stackoverflow.com/a/23063481/10321531
        self.SVMs = SGDClassifier(loss="hinge", penalty="l2", warm_start=True, n_jobs=2)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.D is None:
            self.D = self._initialize_d(X, y)
        if self.Z is None:
            self.Z = self._initialize_z(X, y)
        if self.epoch is None:
            self.epoch = 0

        N = X.shape[1]
        while self.epoch < self.max_iteration:
            self.SVMs.fit(self.Z.T, y.T)
            # D.T@D + e1*I + e (1 + 1/N)
            base = self._d_dagger() + self.reg.e * (1 + 1 / N)
            for i in tqdm(range(N), desc="update Z"):
                Z_except_i = np.delete(self.Z, i, axis=1)
                y_except_i = np.delete(y, i, axis=0)
                m_yi = np.mean(Z_except_i[:, y_except_i == y[i]], axis=1)
                m = np.mean(Z_except_i, axis=1)
                U_zi = self._probable_classes(y, i)
                update_mat = self._get_inv_matrix(base, y, i, U_zi)
                update_x = self.D.T @ X[:, i] + self.reg.e * (2 * m_yi - m)
                if U_zi:
                    svm_regularization = self.SVMs.coef_[U_zi] - self.SVMs.coef_[y[i]]
                    svm_b = (
                        self.SVMs.intercept_[U_zi] - self.SVMs.intercept_[y[i]] + EPS
                    )
                    svm_sum = np.sum(svm_regularization * svm_b.reshape(-1, 1), axis=0)
                    update_x -= self.reg.g * svm_sum
                self.Z[:, i] = update_mat @ update_x

            self._update_dictionary_learning()
            self.dictionary_learning.fit(X.T)
            self.D = self.dictionary_learning.components_.T
            self.epoch += 1

        return self

    def predict(self, X):
        X = np.linalg.inv(self._d_dagger()) @ self.D.T @ X
        return self.SVMs.predict(X.T)

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def save_check_point(self, name: int):
        np.savez_compressed(
            f"./checkpoint/{name}.npz", D=self.D, Z=self.Z, epoch=self.epoch,
        )

    def load_from_checkpoint(self, name: str, y: np.ndarray = None):
        mat = np.load(f"./checkpoint/{name}.npz")
        self.D = mat["D"]
        self.Z = mat["Z"]
        self.epoch = mat["epoch"]
        if y is not None:
            self.SVMs.fit(self.Z.T, y.T)

    def _initialize_d(self, X: np.ndarray, y: np.ndarray):
        n_components = self.K / len(self.classes_)
        assert (
            self.K % len(self.classes_) == 0
        ), f"SAME ATOMS FEATURE COUNT {self.K}/{len(self.classes_)}"

        D = None
        for c in tqdm(range(len(self.classes_)), desc="D-init"):
            Xc = X[:, y == c].T
            assert (
                np.min(Xc.shape) >= n_components
            ), f"PCA COMPONENT ERROR [svd_solver=full] min(X[y=={c}].shape) >= {self.K}/{len(self.classes_)}"
            pca_components = PCA(n_components=int(n_components)).fit(Xc).components_
            if D is None:
                D = pca_components
            else:
                D = np.vstack((D, pca_components))
        return np.array(D).T

    def _initialize_z(self, X: np.ndarray, y: np.ndarray):
        base = self._d_dagger() + self.reg.e * (1 + 1 / X.shape[1])
        N = X.shape[1]
        Z = np.empty((self.K, N))
        for i in tqdm(range(N), desc="Z-init"):
            Z[:, i] = self._get_inv_matrix(base, y, i) @ (
                self.D.T @ X[:, i] + self.reg.e * 2
            )
        return Z

    def _get_inv_matrix(
        self, base: np.ndarray, y: np.ndarray, i: int, U_zi: list = None
    ):
        yi = y[i]
        P = base - self.reg.e * 2 / np.sum(y == yi)
        if not U_zi:  # empty or None
            return np.linalg.inv(P)

        Q = np.power(self.SVMs.coef_[U_zi] - self.SVMs.coef_[yi], 2)
        Q = P + self.reg.g * np.sum(Q)
        return np.linalg.inv(Q)

    def _d_dagger(self):
        base = self.D.T @ self.D + self.reg.e1 * np.eye(self.K)
        return base

    def _svm_decision_function(self, x: np.ndarray, i: int):
        return x.dot(self.SVMs.coef_[i]) + self.SVMs.intercept_[i]

    def _probable_classes(self, y: np.ndarray, i: int):
        zi = self.Z[:, i]
        yi = y[i]
        predict_prob_yi = self._svm_decision_function(zi, yi) + EPS
        U_zi = [
            c
            for c in range(len(self.classes_))
            if c != yi and self._svm_decision_function(zi, c) > predict_prob_yi
        ]
        return U_zi

    def _update_dictionary_learning(self):
        self.dictionary_learning = DictionaryLearning(
            n_jobs=2,
            max_iter=50,
            verbose=True,
            n_components=self.K,
            transform_max_iter=10,
            transform_algorithm="lasso_lars",
            dict_init=self.D.T if self.D is not None else None,
            code_init=self.Z.T if self.Z is not None else None,
        )
