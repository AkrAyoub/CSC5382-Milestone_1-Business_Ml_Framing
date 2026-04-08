from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticTrainingConfig:
    learning_rate: float
    epochs: int
    l2_strength: float
    beta1: float
    beta2: float
    epsilon: float


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fit.")
        return (x - self.mean_) / self.scale_


class NumpyLogisticRegression:
    def __init__(self, config: LogisticTrainingConfig) -> None:
        self.config = config
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        n_samples, n_features = x.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0

        # Keep the Adam state explicit here so the baseline remains easy to
        # inspect without relying on a framework.
        m_w = np.zeros_like(self.weights_)
        v_w = np.zeros_like(self.weights_)
        m_b = 0.0
        v_b = 0.0

        for epoch in range(1, self.config.epochs + 1):
            logits = x @ self.weights_ + self.bias_
            probs = self._sigmoid(logits)

            error = probs - y
            grad_w = (x.T @ error) / n_samples + self.config.l2_strength * self.weights_
            grad_b = float(error.mean())

            m_w = self.config.beta1 * m_w + (1.0 - self.config.beta1) * grad_w
            v_w = self.config.beta2 * v_w + (1.0 - self.config.beta2) * (grad_w ** 2)
            m_b = self.config.beta1 * m_b + (1.0 - self.config.beta1) * grad_b
            v_b = self.config.beta2 * v_b + (1.0 - self.config.beta2) * (grad_b ** 2)

            m_w_hat = m_w / (1.0 - self.config.beta1 ** epoch)
            v_w_hat = v_w / (1.0 - self.config.beta2 ** epoch)
            m_b_hat = m_b / (1.0 - self.config.beta1 ** epoch)
            v_b_hat = v_b / (1.0 - self.config.beta2 ** epoch)

            self.weights_ -= self.config.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.config.epsilon)
            self.bias_ -= self.config.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.config.epsilon)

            loss = -np.mean(y * np.log(probs + 1e-12) + (1.0 - y) * np.log(1.0 - probs + 1e-12))
            loss += 0.5 * self.config.l2_strength * float(np.sum(self.weights_ ** 2))
            self.loss_history_.append(float(loss))

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model has not been fit.")
        return self._sigmoid(x @ self.weights_ + self.bias_)
