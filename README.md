# Neural Network From Scratch

A NumPy-only implementation of a feedforward neural network, built to demonstrate the
core mechanics of deep learning — forward propagation, backpropagation, and
gradient-based optimization — without relying on high-level machine learning frameworks.

| | |
|---|---|
| **Status** | In development (pre-implementation; architecture and roadmap defined) |
| **Type** | Educational / portfolio project (single developer) |
| **Language** | Python 3.10+ |
| **Core dependency** | NumPy (no TensorFlow, PyTorch, or scikit-learn for model logic) |

---

## 1. Project Overview

This project implements a multilayer perceptron (MLP) from first principles using only
NumPy for numerical computation. Every component that a framework would normally
abstract away — layer forward passes, activation derivatives, the backpropagation
algorithm, loss gradients, and the parameter update rules of common optimizers — is
written and verified explicitly.

The objective is **mechanistic understanding, not framework fluency**. High-level
libraries (PyTorch, TensorFlow, Keras, scikit-learn) are deliberately excluded from the
model implementation so that no part of the learning algorithm is hidden behind an
abstraction. NumPy is used solely as a vectorized linear-algebra backend, occupying the
same role that BLAS occupies inside a production framework.

The reference task is handwritten-digit classification on the MNIST dataset, chosen
because it is small enough to train on a CPU yet rich enough to exercise a full
training and evaluation pipeline.

---

## 2. Objectives

This project is intended to demonstrate:

- **Mathematical comprehension** of the forward and backward passes, derived and
  implemented by hand rather than delegated to autograd.
- **Correct gradient computation**, validated against numerical (finite-difference)
  gradients to a defined tolerance.
- **Optimizer internals**, implemented incrementally from plain SGD through to Adam, so
  that the contribution of each algorithmic refinement (mini-batching, momentum,
  adaptive learning rates) is isolated and observable.
- **A reproducible training pipeline** — data loading, preprocessing, training,
  evaluation, and visualization — assembled from the components above.

**Target outcome:** a plain fully-connected MLP achieving approximately **95–98% test
accuracy** on MNIST, trained from scratch on a CPU within a practical time budget.

---

## 3. Requirements

### 3.1 Runtime

| Requirement | Version / Source |
|---|---|
| Python | 3.10 or later |
| NumPy | Core numerical computation |
| Matplotlib | Training curves, confusion matrix, sample visualizations |
| pytest | Unit and gradient-check test suite (development only) |

### 3.2 Dataset

- **MNIST** handwritten digits (60,000 training / 10,000 test images, 28×28 grayscale).
- Sourced via a download utility in the data module (e.g. the canonical Yann LeCun
  distribution or an equivalent mirror). Raw files are stored under `data/` and are
  **not** committed to version control.

### 3.3 Environment Setup

```bash
# Create and activate an isolated environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

All dependencies are pinned in `requirements.txt`. The `data/` directory, virtual
environment, caches, and generated artifacts are excluded from version control via
`.gitignore`.

---

## 4. Architecture Overview

### 4.1 Network Structure

A baseline fully-connected feedforward network:

```
Input (784)  →  Dense(128) + ReLU  →  Dense(64) + ReLU  →  Dense(10) + Softmax
   28×28           hidden layer 1        hidden layer 2       class scores
```

- **Input layer:** 784 units (28×28 pixels flattened).
- **Hidden layers:** 128 and 64 units, ReLU activation.
- **Output layer:** 10 units (one per digit class), softmax activation.
- **Loss:** categorical cross-entropy.

The layer widths are configurable; the values above define the reference architecture
used for the target-accuracy benchmark.

### 4.2 Module Breakdown

| Module | Responsibility |
|---|---|
| `data/` | MNIST download, parsing, normalization, one-hot encoding, train/val/test split, mini-batch iteration |
| `activations.py` | Activation functions and their derivatives (ReLU, sigmoid, softmax) |
| `layers.py` | Dense layer: parameter initialization, forward pass, cached activations for backprop |
| `loss.py` | Cross-entropy loss and its gradient w.r.t. the network output |
| `network.py` | Layer composition, forward orchestration, backpropagation driver |
| `optimizers.py` | Parameter update rules: SGD, mini-batch SGD, Momentum, Adam |
| `train.py` | Training loop: epochs, batching, loss/accuracy logging, checkpointing |
| `evaluate.py` | Test-set evaluation, accuracy, confusion matrix, misclassification inspection |
| `gradient_check.py` | Numerical gradient verification utility |

### 4.3 Repository Layout

```
Neural_Network/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                  # Implementation
│   ├── data/             # Dataset loading and preprocessing
│   ├── activations.py
│   ├── layers.py
│   ├── loss.py
│   ├── network.py
│   ├── optimizers.py
│   ├── gradient_check.py
│   ├── train.py
│   └── evaluate.py
├── tests/                # Unit tests and gradient checks
├── notebooks/            # Exploratory analysis and experiments
├── data/                 # MNIST files (gitignored)
└── docs/                 # Figures, derivation notes, results
```

---

## 5. Development Phases

Development follows a **phase-gated** model. Each phase has an explicit objective, a
task list, defined deliverables, and **exit criteria** — a verification step that must
pass before the next phase begins. This prevents compounding errors: a defect in the
forward pass or gradient computation is caught at its own gate rather than surfacing as
an unexplained training failure several phases later.

### Phase 0 — Environment & Data Pipeline Setup

- **Objective:** Establish a reproducible environment and a working MNIST data pipeline.
- **Tasks:**
  - Initialize repository structure, `requirements.txt`, and `.gitignore`.
  - Implement MNIST download, parsing, and normalization to `[0, 1]`.
  - Implement one-hot label encoding and train/validation/test splitting.
  - Implement a mini-batch iterator.
- **Deliverables:** `src/data/` loader module; reproducible environment setup.
- **Exit criteria:** Loader returns correctly shaped arrays (`X: [N, 784]`,
  `y: [N, 10]`); pixel values normalized; a sample batch renders as recognizable digits.

### Phase 1 — Math Foundations & Gradient-Checking Utility

- **Objective:** Build the numerical tooling that validates all subsequent gradient
  derivations.
- **Tasks:**
  - Implement a finite-difference numerical gradient estimator.
  - Define the relative-error comparison metric between analytic and numerical
    gradients.
  - Establish the acceptance tolerance.
- **Deliverables:** `src/gradient_check.py`.
- **Exit criteria:** Utility correctly verifies the gradient of a known closed-form test
  function (e.g. a quadratic) with relative error below **1e-7**.

### Phase 2 — Forward Propagation

- **Objective:** Implement a correct forward pass through the full network.
- **Tasks:**
  - Implement dense-layer parameter initialization (e.g. He initialization for ReLU).
  - Implement ReLU, sigmoid, and softmax (numerically stable).
  - Compose layers into a forward pass producing class probabilities.
- **Deliverables:** `activations.py`, `layers.py`, forward path in `network.py`.
- **Exit criteria:** Forward pass on a batch returns a valid probability distribution
  (each row sums to 1, all entries in `[0, 1]`); output shape is `[N, 10]`.

### Phase 3 — Loss Function

- **Objective:** Implement categorical cross-entropy and its gradient.
- **Tasks:**
  - Implement numerically stable cross-entropy loss.
  - Derive and implement the loss gradient w.r.t. the softmax input.
- **Deliverables:** `loss.py`.
- **Exit criteria:** Loss is non-negative and behaves correctly at boundary cases
  (near-zero for confident-correct predictions, large for confident-wrong); the
  loss gradient passes the Phase 1 gradient check.

### Phase 4 — Backpropagation

- **Objective:** Implement the full backward pass and verify every gradient.
- **Tasks:**
  - Implement activation derivatives.
  - Implement layer-wise gradient computation (weights, biases, upstream gradient).
  - Wire the backward pass through `network.py`.
- **Deliverables:** Backpropagation driver in `network.py`.
- **Exit criteria:** Analytic gradients for **all** parameters match numerical gradients
  with relative error below **1e-5** on a small network and batch.

### Phase 5 — Optimizers

- **Objective:** Implement parameter-update strategies incrementally and confirm each
  improves convergence over its predecessor.
- **Tasks:**
  - Implement, in order: vanilla SGD → mini-batch SGD → SGD with Momentum → Adam.
  - Expose a uniform optimizer interface so the training loop is optimizer-agnostic.
- **Deliverables:** `optimizers.py`.
- **Exit criteria:** Each optimizer reduces training loss on a small subset over a fixed
  number of steps; Adam converges in fewer epochs than plain SGD on the same problem.

### Phase 6 — Full Training Loop on MNIST

- **Objective:** Integrate all components into an end-to-end training run.
- **Tasks:**
  - Implement the epoch/batch training loop with loss and accuracy logging.
  - Add validation-set evaluation per epoch.
  - Add basic checkpointing of learned parameters.
- **Deliverables:** `train.py`; saved model parameters.
- **Exit criteria:** Training loss decreases monotonically (in trend) across epochs and
  validation accuracy exceeds **90%**, confirming the integrated pipeline learns.

### Phase 7 — Evaluation, Visualization & Documentation Polish

- **Objective:** Measure final performance and document results.
- **Tasks:**
  - Evaluate on the held-out test set.
  - Generate training/validation loss and accuracy curves.
  - Generate a confusion matrix and inspect representative misclassifications.
  - Finalize the Results section and code documentation.
- **Deliverables:** `evaluate.py`; figures in `docs/`; completed Results section.
- **Exit criteria:** Test accuracy meets the **95–98%** target; all figures generated and
  documented.

### Phase 8 — Convolutional Layer From Scratch (Stretch)

- **Objective:** Extend the framework with a hand-implemented convolutional layer.
- **Tasks:**
  - Implement 2D convolution forward and backward passes (naïve, then vectorized).
  - Implement a pooling layer.
  - Train a small CNN and compare against the MLP baseline.
- **Deliverables:** Convolution and pooling modules; comparative results.
- **Exit criteria:** Convolution gradients pass the Phase 1 gradient check; the CNN
  matches or exceeds MLP test accuracy.

---

## 6. Testing & Validation Strategy

Verification is integral to each phase rather than a final-stage activity.

- **Numerical gradient checking.** Every analytically derived gradient (loss, each
  activation, each layer) is compared against a finite-difference estimate. A gradient
  is accepted only when the relative error falls below the phase-specific tolerance
  (1e-7 for the utility's self-test, 1e-5 for backpropagation). This is the primary
  correctness guarantee for the learning algorithm.
- **Unit tests.** Activation functions, their derivatives, and the loss function are
  covered by `pytest` unit tests using small hand-computed reference values and
  boundary cases (e.g. softmax numerical stability on large inputs).
- **Accuracy tracking.** Validation and test accuracy are logged per phase from Phase 6
  onward, providing an end-to-end behavioral check that complements the component-level
  gradient and unit tests.

---

## 7. Engineering Notes

Development follows an incremental, phase-gated approach: each phase is validated —
through numerical gradient checks, unit tests, or accuracy thresholds — before the next
phase begins. This staged verification localizes defects to the component that
introduced them and keeps each gate independently auditable.

An AI coding assistant was used throughout for planning, structuring the development
roadmap, code review, and documentation support. All design decisions, derivations, and
verification criteria were defined and reviewed deliberately; the assistant served as a
planning and review aid rather than a substitute for understanding the underlying
mathematics.

---

## 8. Results

Final numbers are reproduced end-to-end by
`python -m src.evaluate --retrain --epochs 8 --seed 0`. The trained network
is 784 → 128 (ReLU) → 64 (ReLU) → 10 (softmax) with mean-reduction
categorical cross-entropy loss and Adam optimization.

| Metric | Value |
|---|---|
| Test accuracy | **97.75%** (9,775 / 10,000 correct) |
| Test loss (mean cross-entropy) | 0.0799 |
| Final training loss | 0.0310 |
| Final validation accuracy | 97.23% |
| Epochs trained | 8 |
| Optimizer / configuration | Adam, lr = 1e-3, batch size = 128, seed = 0 |

Test accuracy of **97.75%** lands inside the 95–98% target range set in
§5, Phase 7. Training loss decreases monotonically across the 8 epochs
(0.349 → 0.141 → 0.099 → 0.074 → 0.057 → 0.048 → 0.037 → 0.031) and
validation accuracy climbs from 94.33% after epoch 1 to 97.45% at
epoch 7 before settling at 97.23%; the small final-epoch dip is within
the noise expected from mini-batch SGD-style updates.

### Figures

- **Training curves:** [`docs/training_curves.png`](docs/training_curves.png)
- **Confusion matrix:** [`docs/confusion_matrix.png`](docs/confusion_matrix.png)
- **Representative misclassifications:** [`docs/misclassifications.png`](docs/misclassifications.png)

The per-epoch loss and accuracy history used to render the training
curves is persisted at `docs/training_history.json`, so the figures are
regenerable without retraining.

### CNN vs MLP Comparison (Phase 8, stretch)

The Phase 8 stretch goal replaces the dense hidden stack with a small
CNN built from a hand-written 2D convolutional layer (`src/conv.py`)
and max-pooling layer (`src/pooling.py`), both gradient-checked below
`1e-5` end-to-end (Phase 4 tolerance). Architecture:

```
Input [N, 1, 28, 28]
  -> Conv2D(1, 8, 3, padding=1) + ReLU + MaxPool2D(2)   -> [N, 8, 14, 14]
  -> Conv2D(8, 16, 3, padding=1) + ReLU + MaxPool2D(2)  -> [N, 16, 7, 7]
  -> Flatten + Dense(784, 10) + Softmax                 -> [N, 10]
```

Same MNIST split, same Adam optimizer, same `lr = 1e-3`,
`batch_size = 128`, `seed = 0`. Trained for 4 epochs
(reproduced by `python -m src.train_cnn --epochs 4 --seed 0`):

| Model | Params | Epochs | Test accuracy | Test loss |
|---|---|---|---|---|
| MLP (baseline) | ~109k | 8 | 97.75% | 0.0799 |
| CNN (Phase 8) | ~9.1k | 4 | **98.03%** | 0.0597 |

The CNN meets the "matches or exceeds MLP test accuracy" Phase 8 exit
criterion (98.03% ≥ 97.75%) with roughly one-twelfth the parameter
count and half the training epochs. Every convolutional parameter
(kernels and biases) passes the Phase 1 numerical gradient check at
tolerance `1e-5` — the highest per-parameter relative error observed
across the CNN was `3.05e-10`, five orders below tolerance.

---

## 9. References

- Michael Nielsen, *Neural Networks and Deep Learning* —
  <http://neuralnetworksanddeeplearning.com/>
- Stanford CS231n, *Convolutional Neural Networks for Visual Recognition* (course notes) —
  <https://cs231n.github.io/>
- 3Blue1Brown, *Neural Networks* series —
  <https://www.3blue1brown.com/topics/neural-networks>
