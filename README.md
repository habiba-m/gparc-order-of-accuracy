# Analyzing the Order of Accuracy of G-PARC

**Habiba Morsy, Tyler Abele — University of Virginia, School of Data Science, Spring 2026**

We measure the empirical order of accuracy of [G-PARC](https://github.com/JackBeerman/G-PARC), a physics-aware graph neural network PDE solver, on 2D linear advection. The core question: do neural PDE solvers remain stable because they implicitly behave like low-order dissipative schemes?

---

## Repository Structure

```
.
├── generate_data.py       # generates all training and evaluation data
├── train.py               # trains G-PARC on gaussian or disc problem
├── evaluate.py            # convergence study: error vs mesh spacing
├── data/
│   ├── gaussian/
│   │   ├── train/         # 50 simulations, N=64
│   │   ├── val/           # 10 simulations, N=64
│   │   ├── test/          # 10 simulations, N=64
│   │   └── eval/          # res_64, res_48, res_32, res_24, res_16
│   └── disc/
│       └── ...            # same structure, 30 train sims
├── outputs/
|  ├── gaussian/
|  │   └── best_model.pth
|  └── disc/
|      └── best_model.pth
├── figures/
|  └── convergence_2d.png
|  └── convergence_disc.png
|  └── dataset_figure.png
|  └── results_2d.csv
|  └── results_disc.csv
├── numerical/
   └── convergence.py    # Computes L2 error vs mesh spacing for
   └── plot.py           # Produce all figures used in the report
   └── reference.py      # Reference solutions for the linear advection equation
```

---

## Dependencies

```bash
pip install torch torch_geometric numpy matplotlib
```

G-PARC source code is required and must be cloned separately:

```bash
git clone https://github.com/JackBeerman/G-PARC ~/G-PARC
```

All scripts automatically add `~/G-PARC` to the Python path.

---

## Reproducing Results

### Step 1 — Generate Data

```bash
python generate_data.py
```

Generates training, validation, test, and evaluation splits for both
the Gaussian (smooth) and disc (non-smooth) problems. Training data
uses N=64 (4096 nodes). Evaluation data covers 5 resolutions:
N=64, 48, 32, 24, 16. Takes ~5 minutes.

### Step 2 — Train Models

Train the Gaussian model (change `PROBLEM = "gaussian"` at the top of `train.py`):

```bash
python train.py
```

Train the disc model (change `PROBLEM = "disc"` at the top of `train.py`):

```bash
python train.py
```

Or submit both as SLURM jobs in parallel:

```bash
sbatch train_gaussian.slurm
sbatch train_disc.slurm
```

Training takes 3–6 hours per model on a single GPU. Checkpoints are
saved to `outputs/gaussian/best_model.pth` and `outputs/disc/best_model.pth`.

### Step 3 — Evaluate

```bash
python evaluate.py
```

Runs the convergence study: for each resolution, feeds the exact state
and predicts one step forward, computes Rel-L2 and L∞ error against the
analytical solution, and fits `log(error) vs log(h)` to extract the
empirical order of accuracy. Saves `outputs/convergence_plot.png`.

---

## Key Design Choices

**Why N=64 for training?** G-PARC's MLS operators are mesh-invariant —
they recompute stencil weights from local geometry at inference time.
The model trains at N=64 and is evaluated at coarser resolutions without
retraining. Going finer than the training resolution breaks the learned
feature representations.

**Why single-step evaluation?** The convergence study measures spatial
accuracy, not accumulated rollout error. Feeding the exact state at each
step isolates the model's one-step truncation error, which is the correct
quantity for order-of-accuracy analysis.

**Why delta supervision for Gaussian?** Each step changes u by ~0.001.
Supervising on the increment rather than the absolute value amplifies
the learning signal ~100×. For the disc, boundary nodes already have
Δu = ±1 so delta supervision is not needed and causes instability.

---

## Results Summary

| Problem  | Rel-L2 slope | L∞ slope | Interpretation             |
| -------- | ------------ | -------- | -------------------------- |
| Gaussian | 1.22         | 1.06     | ~first-order (dissipative) |
| Disc     | 0.75         | 0.04     | sub-first-order, flat L∞   |

G-PARC on smooth fields converges at ~first-order, below the MLS
theoretical consistency order of ~2. The neural components (graph
attention, SPADE fusion) introduce implicit dissipation that degrades
accuracy. On the non-smooth disc, the flat L∞ slope indicates the
model smears the discontinuity at every resolution.
