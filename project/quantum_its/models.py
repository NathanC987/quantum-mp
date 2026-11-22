"""Models: variational quantum classifier (VQC), quantum-kernel SVM and classical baselines."""
import time
import warnings

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .circuits import (N_QUBITS, CircuitSpec, expval_z, probabilities, run_circuit, ry,
                       ring_pairs, zero_state, zz_coefficient, zz_phase)

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", category=FutureWarning)


class VQC:
    """Hybrid classifier: P(congested | x) = sigmoid(a * <Z_0> + b).

    Quantum part: |Phi(x; theta)> = prod_l [U(theta_l) U_phi(x) U_encode(w_l * x)] |0000>
    Classical part: the measured expectation <Z_0> is rescaled by (a, b) and thresholded at 0.5.
    """

    def __init__(self, spec: CircuitSpec, seed: int = 0):
        self.spec = spec
        g = torch.Generator().manual_seed(seed)
        self.theta = (torch.rand((spec.layers, spec.ansatz_reps, N_QUBITS, 3), generator=g) * 2 - 1) * np.pi
        self.w = torch.ones((spec.layers, N_QUBITS))
        self.ab = torch.tensor([3.0, 0.0])
        self.history = []

    def parameters(self):
        return [self.theta, self.w, self.ab] if self.spec.scaling else [self.theta, self.ab]

    def expval(self, X):
        X = torch.as_tensor(X)
        return expval_z(run_circuit(self.spec, X, self.theta, self.w), 0)

    def logits(self, X):
        return self.ab[0] * self.expval(X) + self.ab[1]

    def fit(self, X, y, epochs=150, lr=0.05):
        X, y = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float64)
        for p in self.parameters():
            p.requires_grad_(True)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        t0 = time.perf_counter()
        for _ in range(epochs):                                   # full-batch Adam
            opt.zero_grad()
            loss = loss_fn(self.logits(X), y)
            loss.backward(); opt.step()
            self.history.append(loss.item())
        self.fit_seconds = time.perf_counter() - t0
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            return torch.sigmoid(self.logits(X)).numpy()

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def state_probabilities(self, x):
        """Measurement distribution over the 16 basis states for one snapshot (for the dashboard)."""
        with torch.no_grad():
            psi = run_circuit(self.spec, torch.as_tensor(np.atleast_2d(x)), self.theta, self.w)
            return probabilities(psi)[0].numpy()

    def state_dict(self):
        return dict(theta=self.theta.numpy(), w=self.w.numpy(), ab=self.ab.numpy(), spec=self.spec.describe())

    @classmethod
    def from_state(cls, d):
        s = d["spec"]
        m = cls(CircuitSpec(s["layers"], s["feature_map"], s["entangle"], s["scaling"], s["ansatz_reps"]))
        m.theta, m.w, m.ab = (torch.as_tensor(d[k]) for k in ("theta", "w", "ab"))
        return m


# ---------------- quantum kernel ----------------
def feature_map_states(X, reps=2):
    """|phi(x)> = [U_phi(x) U_encode(x)]^reps |0000>  (ZZ feature map, ring pairs)."""
    X = torch.as_tensor(X)
    psi = zero_state(X.shape[0])
    for _ in range(reps):
        for k in range(N_QUBITS):
            psi = ry(psi, k, X[:, k])
        for k, m in ring_pairs():
            psi = zz_phase(psi, k, m, zz_coefficient(X[:, k], X[:, m]))
    return psi.reshape(X.shape[0], -1)


def quantum_kernel(A, B, reps=2):
    """Fidelity kernel K(x, x') = |<phi(x)|phi(x')>|^2."""
    sa, sb = feature_map_states(A, reps), feature_map_states(B, reps)
    return (sa.conj() @ sb.T).abs().pow(2).numpy()


class QuantumKernelSVM:
    """SVM on the fidelity kernel. The feature-map input is scaled by a bandwidth b (x -> b*x),
    chosen with C by 3-fold cross-validation on the training set only."""

    def __init__(self, reps=2, bandwidths=(0.1, 0.25, 0.5, 1.0)):
        self.reps, self.bandwidths = reps, bandwidths

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        best = None
        for bw in self.bandwidths:
            K = quantum_kernel(bw * self.X_train, bw * self.X_train, self.reps)
            gs = GridSearchCV(SVC(kernel="precomputed", probability=True, random_state=0),
                              {"C": [0.1, 1, 10, 100]}, cv=3).fit(K, y)
            if best is None or gs.best_score_ > best[0]:
                best = (gs.best_score_, bw, gs.best_estimator_)
        _, self.bandwidth, self.svc = best
        return self

    def predict_proba(self, X):
        b = self.bandwidth
        return self.svc.predict_proba(quantum_kernel(b * np.asarray(X), b * self.X_train, self.reps))[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


# ---------------- classical baselines (same 4 features, same training samples) ----------------
def classical_models():
    return {
        "Logistic regression": lambda: LogisticRegression(max_iter=2000),
        "SVM (RBF)": lambda: SVC(probability=True, random_state=0),
        "Random forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
        "MLP (16-16)": lambda: MLPClassifier((16, 16), max_iter=2000, random_state=0),
    }


def classical_inputs(X):
    """Classical models get the cyclic hour as (sin, cos) so they are not disadvantaged."""
    X = np.asarray(X)
    return np.column_stack([np.sin(X[:, 0]), np.cos(X[:, 0]), X[:, 1:]])


def metrics(y, proba):
    pred = (proba > 0.5).astype(int)
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred), recall=recall_score(y, pred),
                auc=roc_auc_score(y, proba))
