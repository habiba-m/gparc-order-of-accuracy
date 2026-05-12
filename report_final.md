# Analyzing the Order of Accuracy of Neural PDE solvers Using a Graph-Based PARC (G_PARC)

**Habiba Morsy, Tyler Abele**
University of Virginia, School of Data Science
PADL 2026 — Final Project
---

## 1. Problem Definiton

Classical numerical PDE solvers come with well-understood guarantees: A finite-difference or finit-volume scheme has a provable order of accuracy. This means that the error decays as $\mathcal{0}(h^p)$ as the mesh spacing $h$ shrinks. Additionally, the stability is enforced explicitly through conditions such as the CFL constraint and entropy stability. These aspects guarantees explain why classical solvers remain bound over long time integrations even though discretizations introduces error at every step.

Neural-network based PDE solvers typically inherit none of these guarantees explicitly. They are trained in a way to minimize loss on solution snapshots. Furthermore, there is no built-in CFL check, no entropy condition, and no formal convergence theorem. However, in practice, they routinely remain bounded and qaulitatively correct over long rollouts. This raises a question:

1. On **smooth** problems, the neural solver should exhibit a measurable, finite order of accuracy $p$ when error is plotted against mesh spacing on a log–log scale.
2. On problems with **discontinuities**, the order of accuracy should degrade toward the first-order behavior characteristic of dissipative schemes (e.g., first-order upwind) at shocks and contact discontinuities.

### Governing equations
 
We study the **linear advection equation**, the simplest non-trivial hyperbolic PDE and the standard testbed for convergence analysis because its exact solution is known analytically.
 
2D form (smooth test):
$$u_t + \mathbf{v} \cdot \nabla u = 0, \qquad u(\mathbf{x}, 0) = \exp\!\left(-\tfrac{(x-x_0)^2 + (y-y_0)^2}{\sigma^2}\right)$$
 
1D form (non-smooth test):
$$u_t + c\, u_x = 0, \qquad u(x,0) = \mathbb{1}_{[a,b]}(x) \quad \text{(tophat)}$$
 
The analytical solution is pure translation of the initial condition along the velocity vector, so the exact reference $u^*(\mathbf{x}, t)$ is available at any resolution. This is what makes a clean convergence study possible.

### Physics / structure encoded by the model
 
Linear advection has three structural properties that a physics-aware model should ideally preserve: **mass / $L^1$ conservation** of the transported quantity, the **maximum principle** ion which the solution is bounded by the extrema of the initial condition, and **translation invariance** in which the operator is local and homogeneous in space. PARC's recurrent integrator structure (predict the time derivative, then integrate) encodes locality naturally, and the MLS-based graph derivative operators in G-PARC give us a discretization that is consistent on arbitrary meshes — which is what makes resolution-independent evaluation possible at all.
 
---

## 2. Data

### Generation

All training and test data are generated synthetically by solving the advection PDE on a high-resolution uniform grid with a stable classical solver [**TODO: confirm — Lax-Wendroff or RK4+WENO**]. Because the equation is linear and the exact solution is known, we additionally generate *analytical* reference data on the fly at every evaluation resolution. This analytical reference is what we use for the convergence-error measurement, eliminating numerical-reference error as a confound.

### Parameters varied

| Parameter | Symbol | Value / Range |
|---|---|---|
| Gaussian center | $(x_0, y_0)$ | sampled in $[-0.5, 0.5]^2$ |
| Gaussian width | $\sigma$ | $0.15$ |
| Velocity vector | $\mathbf{v}$ | $(1.0, 0.5)$, periodic domain $[-1, 1]^2$ |
| Time horizon | $T$ | $0.5$ |
| Training resolution | $h_{\text{train}}$ | $N = $ [**TODO**] |
| Eval resolutions (2D) | $h_{\text{eval}}$ | $N \in \{128, 96, 64, 48, 32\}$ |
| Eval resolutions (1D) | $h_{\text{eval}}$ | $N \in \{512, 256, 128, 64, 32\}$ |

### Dataset size

[**TODO** — approximately N trajectories for training, M held-out.] The headline convergence-study evaluation uses the analytical solution rather than held-out trajectories, so the dataset of record for that result is the set of eval resolutions listed above.

### Visualization

**Figure 1** (`figures/qualitative_2d.png` / `qualitative_1d.png`) shows snapshots of the 2D Gaussian and 1D tophat advection problems, with the analytical reference and the dissipative behavior of first-order upwind side by side. The progressive blurring of upwind is the visual signature of numerical dissipation, which is precisely the behavior we hypothesize G-PARC will exhibit implicitly.

---
## 3. Method
 
### Choice of method: G-PARC
 
We extend the **PARC** framework (Nguyen et al., 2024), Physics-Aware Recurrent Convolutional networks to operate on graphs rather than fixed regular grids. The graph variant, **G-PARC** (Beerman et al., 2026), replaces the convolutional spatial operators with derivative operators built from **moving least squares (MLS)** on an arbitrary point cloud or mesh.
 
The methodological reason for choosing G-PARC over a vanilla CNN-PARC or a Fourier neural operator is decisive for *this* study: a convergence analysis requires evaluating the *same trained model* at progressively coarser resolutions, which a fixed-grid CNN cannot do without retraining or interpolation tricks that themselves introduce error and contaminate the slope measurement. G-PARC's MLS-graph operators are mesh-agnostic by construction meaning the same learned weights act on any point cloud. In theory any error trend observed across resolutions is attributable to the learned dynamics, not to a resolution-dependent architecture.