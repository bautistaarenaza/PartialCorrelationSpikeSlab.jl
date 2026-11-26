# PartialCorrelationSpikeSlab.jl

[![Build Status](https://github.com/bautistaarenaza/PartialCorrelationSpikeSlab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bautistaarenaza/PartialCorrelationSpikeSlab.jl/actions/workflows/CI.yml?query=branch%3Amain)

PartialCorrelationSpikeSlab.jl implements the partial correlation spike-and-slab (PCSS) estimator of a sparse Gaussian graphical model, the graph whose edges are the nonzero partial correlations between the variables.

The model writes the precision matrix as $\Omega = \Lambda(I - R)\Lambda$, with $\Lambda$ the diagonal matrix of scales and $R$ the matrix of partial correlations.
Each partial correlation gets a spike-and-slab prior: a Laplace spike of inverse width $\gamma$, truncated to $(-1, 1)$, and a uniform slab, mixed with weight $\eta$.
The estimate is a maximum a posteriori fit of $(\Lambda, R, \eta)$, computed by an ECM algorithm along a continuation path in $\gamma$.

The only hyperparameter is $\gamma$.
It can be set to a calibrated value $\gamma_q$, which needs a single fit, or selected by EBIC over a grid.

## Installation

The package requires Julia 1.10 or later.

```julia
using Pkg
Pkg.add("PartialCorrelationSpikeSlab")
```

The development version can be installed with `Pkg.add(url = "https://github.com/bautistaarenaza/PartialCorrelationSpikeSlab.jl")`.

## Usage

```julia
using PartialCorrelationSpikeSlab, LinearAlgebra, Random

# A chain graph on p = 20 variables, with partial correlations 0.4
p, n = 20, 400
Omega = Matrix(SymTridiagonal(ones(p), fill(-0.4, p - 1)))
X = randn(Xoshiro(1), n, p) / cholesky(Omega).U'     # n draws from N(0, Omega⁻¹)

fit = pcss_fit(X = X)
```

`pcss_fit` returns a named tuple with

| field | content |
|---|---|
| `fit.R` | estimated partial correlation matrix (zero diagonal) |
| `fit.lambda` | estimated scales, the diagonal of $\Lambda$ |
| `fit.eta` | estimated mixture weight |
| `fit.gamma` | hyperparameter used |

and `pcss_edges(fit)` gives the edges of the estimated graph (see below).

The estimated precision matrix is

```julia
Omega_hat = Diagonal(fit.lambda) * (I - fit.R) * Diagonal(fit.lambda)
```

Instead of the data matrix, `pcss_fit` also accepts the sample size and the sample covariance, `pcss_fit(n = n, S = S)`.
`S` should have divisor `n`, as `cov(X; corrected = false)`.
A divisor `n − 1` leaves `R` and `eta` unchanged and rescales `lambda` by $\sqrt{(n-1)/n}$.
`pcss_fit` throws an `ArgumentError` on missing, NaN or infinite entries, on a constant column of `X`, and on an `S` that is not symmetric positive semidefinite with a positive diagonal.

### Estimated graph

```julia
edges = pcss_edges(fit)
```

returns the edges of the estimated graph as a vector of pairs `(i, j)` with `i < j`.
In the example above, `edges` is exactly the chain `(1, 2), (2, 3), …, (19, 20)`.

The edges are the pairs for which the slab is more probable than the spike, that is, those with $|\hat r_{ij}| > r_c(\hat\eta)$, where

$$
r_c(\eta) = \frac{1}{\gamma}\left[\log\tilde\gamma + \log\frac{1-\eta}{\eta}\right], \qquad \tilde\gamma = \frac{\gamma}{1 - e^{-\gamma}} .
$$

Small nonzero entries of `fit.R` below $r_c(\hat\eta)$ are therefore not edges, so the graph should be read from `pcss_edges` and not from the nonzero entries of `fit.R`.
`pcss_edges` also accepts a named tuple `(R = R, eta = eta, gamma = gamma)`, for estimates stored elsewhere.

### Choosing γ

The keyword `criterion` sets $\gamma$.

* `criterion = :gamma_q` (default) uses the calibrated value `gamma_q(n, p; q, m_max, d_max)`.
  It is the $(1 - q)$-quantile, under a ground truth with no edges, of the limiting distribution of a lower bound on $\gamma$ above which the diagonal estimate is a strict local maximum that beats every estimate supported on a forest with at most `m_max` edges and maximum degree at most `d_max`.
  The defaults are `q = 0.05`, `m_max = 5` (or `p − 1` if smaller) and `d_max = 1`.
* `criterion = :ebic` selects $\gamma$ by the EBIC, with parameter `gamma_ebic = 0.5`, over a grid of `K = 25` values log-spaced from $\gamma_q/8$ to $8\gamma_q$, or over `gamma_grid` when given.
  Each value of the grid needs its own fit.
* `criterion = :bic` selects $\gamma$ by the BIC over the same grid.

A value given as `gamma = ...` overrides `criterion`.

`gamma_q` is computed by Monte Carlo, with `nreps = 20_000` replicates by default.
It depends on the data only through `n` and `p`, so it can be computed once and reused,

```julia
g = gamma_q(n, p; rng = Xoshiro(0))
fit = pcss_fit(X = X, gamma = g)
```

Passing a seeded `rng`, here or to `pcss_fit`, makes the result reproducible.

### Unicode and ASCII names

Several keywords also accept the Unicode names of the paper's notation: `γ` for `gamma`, `m̄` for `m_max`, `d̄` for `d_max`, `γ_ebic` for `gamma_ebic`, `γ0` for `gamma0`, `Ω0` for `Omega0` and `γ_grid` for `gamma_grid`.
`γ_q` is the same function as `gamma_q`.

The remaining options (the continuation path, the starting point and the comparison with the diagonal estimate) are described in the docstring, `?pcss_fit`.

## Citation

A paper describing the method is in preparation.
This section will link to it.

## License

MIT, see [LICENSE](LICENSE).
