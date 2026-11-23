"""
    PCSS

Partial correlation spike-and-slab (PCSS) estimator of a sparse Gaussian
graphical model.

The model has precision matrix `Ω = Λ(I − R)Λ`, with `Λ` the diagonal matrix of
scales and `R` the partial correlation matrix.  Each partial correlation has a
spike-and-slab prior: a Laplace spike of inverse width `γ`, truncated to
`(−1, 1)`, and a uniform slab, mixed with weight `η`.  The PCSS estimate is the
output of the procedure of the paper: ECM fits of the log-posterior

    L(Λ, R, η) = L_s(Λ, R) − Σ_{i<j} pen(r_ij, η),
    L_s(Λ, R)  = −(n/2) tr[S_n Λ(I−R)Λ] + (n−1) log det Λ + (n/2) log det(I−R),
    pen(r, η)  = −log[η + (1−η) γ̃ e^{−γ|r|}],   γ̃ = γ/(1 − e^{−γ}),

along a continuation path in `γ`, followed by a comparison with the diagonal
estimate.

The exported functions are `pcss_fit`, which computes the estimate,
`pcss_edges`, which returns the edges of the estimated graph, and `γ_q` (ASCII
alias `gamma_q`), which computes the calibrated hyperparameter.

# Notation (as in the paper)

    λ  : diagonal of the scale matrix Λ, λ_i = √Ω_ii           (length-p vector)
    R  : partial correlation matrix, zero diagonal
    η  : mixture weight
    γ  : hyperparameter, the inverse width of the spike;  γ̃ = γ/(1 − e^{−γ})
    σ  : slab-membership probabilities computed in the E-step
    ρ  : sample correlation matrix ρ_n
    Σ̂  : Λ ρ Λ, the sample covariance on the scale set by Λ
    Θ  : I − R, the ADMM variable that carries the penalty and the unit diagonal
    Φ  : the ADMM copy of Θ that carries the smooth terms (positive definite)
    A  : ADMM dual variable;  β : ADMM parameter

# Correlation scale

All computations are done on the correlation scale, that is, with `S_n`
replaced by `ρ_n` and `Λ` by the standardized scale `D_s Λ`, where
`D_s = diag(√(S_n)_ii)`.  By scale invariance this changes neither `R` nor `η`,
and the scales are returned to the data scale at the end.
"""
module PCSS

using LinearAlgebra
using LinearAlgebra.LAPACK: syevr!
using Statistics: cor, std, quantile!
using Random: AbstractRNG, default_rng, randexp
using StatsFuns: tdistinvcdf
using FastLapackInterface: HermitianEigenWs

export pcss_fit, pcss_edges, γ_q, gamma_q

"""
    slab_membership!(σ, R, η, γ)

E-step: the slab-membership probabilities

    σ_ij = η / (η + (1−η) γ̃ e^{−γ|r_ij|}) = 1 / (1 + e^{−γ(|r_ij| − r_c(η))}),
    r_c(η) = [log γ̃ + log(1−η) − log η] / γ,

written into `σ` in place, with the diagonal set to 1.  Valid for every
`η ∈ [0, 1]`: `r_c = +Inf` at `η = 0` gives `σ = 0`, and `r_c = −Inf` at `η = 1`
gives `σ = 1`.  The form through `r_c` never multiplies `Inf` by an underflowed
`e^{−γ|r_ij|} = 0`, which would give `NaN`.
"""
function slab_membership!(σ::Matrix{Float64}, R::AbstractMatrix{Float64},
  η::Float64, γ::Float64)
  γ̃ = γ / (1 - exp(-γ))
  rc = (log(γ̃) + log1p(-η) - log(η)) / γ
  @inbounds @. σ = 1 / (1 + exp(-γ * (abs(R) - rc)))
  @inbounds for i in axes(σ, 1)
    σ[i, i] = 1.0
  end
  return σ
end

"""
    logdet_and_quad(λ, R, ρ)

The two terms of the smooth part `L_s` that depend on `R`, on the correlation
scale,

    log det(I − R)   and   λ'(ρ∘(I−R))λ = tr[ρ Λ(I−R)Λ].

The log-determinant comes from a Cholesky factorization of `I − R`, and is
`-Inf` when `I − R` is not positive definite.  Returns `(logdet, quad)`.
"""
function logdet_and_quad(λ::Vector{Float64}, R::AbstractMatrix{Float64},
  ρ::AbstractMatrix{Float64})
  p = length(λ)
  Θ = I - R
  quad = 0.0                                             # λ'(ρ∘Θ)λ
  @inbounds for j in 1:p
    s = 0.0
    for i in 1:p
      s += ρ[i, j] * Θ[i, j] * λ[i]
    end
    quad += s * λ[j]
  end
  C = cholesky!(Symmetric(Θ); check=false)               # overwrites Θ
  ld = issuccess(C) ? logdet(C) : -Inf
  return ld, quad
end

"""
    log_posterior(λ, R, η, n, ρ, γ)

Log-posterior `L` of the paper, divided by `n` and up to an additive constant,
on the correlation scale,

    (1 − 1/n) Σ_i log λ_i + ½[log det(I−R) − λ'(ρ∘(I−R))λ] − (1/n) Σ_{i<j} pen(r_ij, η).

It is `-Inf` when `I − R` is not positive definite.  Used in the stopping rule
of the ECM algorithm and in the comparison with the diagonal estimate.  Each
`−pen(r_ij, η) = log(e^a + e^b)`, with `a = log η` and
`b = log(1−η) + log γ̃ − γ|r_ij|`, is computed as `max(a, b) + log1p(e^{−|a−b|})`,
which stays finite for every `η ∈ [0, 1]` even when `e^{−γ|r_ij|}` underflows.
"""
function log_posterior(λ::Vector{Float64}, R::AbstractMatrix{Float64},
  η::Float64, n::Int, ρ::AbstractMatrix{Float64}, γ::Float64)
  p = size(ρ, 1)
  ld, quad = logdet_and_quad(λ, R, ρ)
  smooth = (1 - 1 / n) * sum(log, λ) + 0.5 * (ld - quad)
  γ̃ = γ / (1 - exp(-γ))
  a = log(η)                                             # −Inf at η = 0
  b0 = log1p(-η) + log(γ̃)                                # −Inf at η = 1
  minus_pen = 0.0                                        # −Σ_{i<j} pen(r_ij, η)
  @inbounds for j in 2:p, i in 1:j-1
    b = b0 - γ * abs(R[i, j])
    minus_pen += max(a, b) + log1p(exp(-abs(a - b)))     # log(e^a + e^b)
  end
  return smooth + minus_pen / n
end

"""
    ADMMWorkspace(Θ0)

ADMM workspace of the R-update, started at `Θ = Θ0`.  `Θ`, `A` and `β` are
carried across ECM iterations and across the stages of the continuation path,
so that every R-update is warm-started.  The buffers and the LAPACK
eigendecomposition workspace are reused to avoid allocation.
"""
mutable struct ADMMWorkspace{E}
  Θ::Matrix{Float64}      # I − R, carries the penalty and the unit diagonal
  A::Matrix{Float64}      # dual variable
  β::Float64              # ADMM parameter (adapted by residual balancing)
  G::Matrix{Float64}      # (n/2)Σ̂ + A − βΘ   (overwritten by the eigendecomposition)
  Φ::Matrix{Float64}      # copy of Θ that carries the smooth terms (positive definite)
  Vw::Matrix{Float64}     # buffer for V diag(√φ)
  eigws::E                # symmetric eigendecomposition (syevr!) workspace
end

function ADMMWorkspace(Θ0::AbstractMatrix{Float64})
  p = size(Θ0, 1)
  Θ = Matrix{Float64}(Θ0)
  G = similar(Θ)
  eigws = HermitianEigenWs(G; vecs=true)
  ADMMWorkspace(Θ, zeros(p, p), 1.0, G, similar(Θ), similar(Θ), eigws)
end

"""
    update_R!(R, n, Σ̂, wts, ws, tol; admm_max_iter = 2^8)

R-update of the M-step: the diagonal-constrained, entrywise-weighted graphical
lasso of the paper, divided by 2,

    min_{Θ : Θ_ii = 1, Θ ≻ 0}  (n/2)[tr(Σ̂Θ) − log det Θ] + Σ_{i≠j} wts_ij |Θ_ij|,

with weights `wts_ij = γ(1 − σ_ij)/2`, solved by ADMM with the copy `Φ` of `Θ`,
the dual variable `A` and the parameter `β`, warm-started through `ws`.  Since
the objective is divided by 2, `n/2` replaces `n` in the ADMM updates of the
paper.  On return, `R = I − Θ`, after `Θ` is shifted so that its smallest
eigenvalue equals that of `Φ`, which is positive, and rescaled to unit
diagonal.  Only the upper triangle of `Σ̂` is read.  Returns the number of ADMM
iterations.
"""
function update_R!(R::Matrix{Float64}, n::Int,
  Σ̂::Matrix{Float64}, wts::Matrix{Float64},
  ws::ADMMWorkspace, tol::Float64; admm_max_iter::Int=2^8)
  p = size(R, 1)
  Θ, A, Φ = ws.Θ, ws.A, ws.Φ
  G, Vw = ws.G, ws.Vw
  β = ws.β
  halfn = 0.5 * n
  mink = 1.0                                   # smallest eigenvalue of the last Φ

  converged = false
  s = 0
  while !converged && s < admm_max_iter
    s += 1

    # ---- Φ-update -----------------------------------------------------
    # stationarity  (n/2)Φ⁻¹ − βΦ = G := (n/2)Σ̂ + A − βΘ,
    # so the eigenvalues of Φ are φ_i = (√(g_i² + 2nβ) − g_i) / (2β).
    @inbounds @. G = halfn * Σ̂ + A - β * Θ
    g, V = syevr!(ws.eigws, 'V', 'A', 'U', G, 0.0, 0.0, 0, 0, -1.0)   # G = V diag(g) V'
    @inbounds @. g = (sqrt(g^2 + 2 * n * β) - g) / (2β)              # g ← eigenvalues of Φ
    mink = max(minimum(g), 2.0^-30)
    @inbounds @. g = sqrt(g)
    mul!(Vw, V, Diagonal(g))
    BLAS.syrk!('U', 'N', 1.0, Vw, 0.0, Φ)     # Φ = (V diag(√φ))(V diag(√φ))', upper triangle
    @inbounds for j in 1:p, i in j+1:p
      Φ[i, j] = Φ[j, i]                       # fill the lower triangle
    end

    # ---- Θ-update (soft-thresholding), dual update and residuals -------
    ϵ1² = 0.0                                 # ‖Φ − Θ‖²       (primal residual)
    ϵ2² = 0.0                                 # ‖Θ − Θ_prev‖²  (dual residual / β)
    nΦ² = 0.0
    nΘ² = 0.0
    nA² = 0.0
    @inbounds for j in 1:p, i in 1:p
      φ = Φ[i, j]
      x = β * φ + A[i, j]
      θ = i == j ? 1.0 :                      # unit diagonal, so R has zero diagonal
          sign(x) * max(abs(x) - wts[i, j], 0.0) / β
      a = A[i, j] + β * (φ - θ)
      ϵ1² += (φ - θ)^2
      ϵ2² += (θ - Θ[i, j])^2
      nΦ² += φ^2
      nΘ² += θ^2
      nA² += a^2
      Θ[i, j] = θ
      A[i, j] = a
    end

    # ---- stopping rule and residual balancing of β ---------------------
    ϵ1 = sqrt(ϵ1²)
    ϵ2 = β * sqrt(ϵ2²)
    converged = (ϵ1 ≤ tol * sqrt(max(nΦ², nΘ²))) && (ϵ2 ≤ tol * sqrt(nA²))
    if ϵ1 > 10ϵ2
      β *= 2
    elseif ϵ2 > 10ϵ1
      β /= 2
    end
  end
  ws.β = β                                      # keep the adapted β for the next R-update

  # ---- from Θ to the partial correlations R ------------------------------
  # Work on a copy, so that the warm-start Θ keeps its unit diagonal.
  P = ws.Φ                                      # Φ is no longer needed; reuse its storage
  copyto!(P, Θ)
  min_Θ = eigmin(Symmetric(P))
  @inbounds for i in 1:p
    P[i, i] += (mink - min_Θ)                 # smallest eigenvalue of P := that of Φ
  end
  # r_ij = −P_ij / √(P_ii P_jj),  r_ii = 0
  d = [sqrt(P[i, i]) for i in 1:p]
  @inbounds for j in 1:p
    for i in 1:p
      R[i, j] = i == j ? 0.0 : -P[i, j] / (d[i] * d[j])
    end
  end
  return s
end

"""
    update_λ!(λ, n, ρR; tol = 2.0^-10, max_iter = 2^10)

Λ-update of the M-step: coordinate descent on the subproblem of the paper, on
the correlation scale (`(S_n)_ii = 1`),

    λ_i ← [√(b_i² + 4(1 − 1/n)) − b_i] / 2,   b_i = −Σ_{j≠i} ρ_ij r_ij λ_j,

each coordinate using the most recent values of the others.  The sweeps stop
when the relative change of `log λ` falls below `tol`.  `ρR = ρ ∘ R` must be
symmetric, since `b_i` is accumulated down column `i`.
"""
function update_λ!(λ::Vector{Float64}, n::Int, ρR::AbstractMatrix{Float64};
  tol::Float64=2.0^-10, max_iter::Int=2^10)
  c = 1.0 - 1 / n
  old = similar(λ)
  for _ in 1:max_iter
    @inbounds @. old = log(λ)
    @inbounds for i in eachindex(λ)
      b = 0.0
      for j in eachindex(λ)
        b -= λ[j] * ρR[j, i]
      end
      λ[i] = 0.5 * (sqrt(b^2 + 4c) - b)
    end
    # relative change on the log scale
    num = 0.0
    den = 0.0
    @inbounds for i in eachindex(λ)
      li = log(λ[i])
      num += (li - old[i])^2
      den += li^2
    end
    sqrt(num) < tol * sqrt(max(den, eps())) && break
  end
  return λ
end

"""
    sum_upper(B)

Sum of the entries of `B` above the diagonal.  Used in the η-update.
"""
@inline function sum_upper(B::AbstractMatrix{Float64})
  s = 0.0
  @inbounds for j in 2:size(B, 1), i in 1:j-1
    s += B[i, j]
  end
  return s
end

"""
    support_change(R, old)

Number of pairs that enter or leave the support of `R` relative to `old`.  Used
in the stopping rule of the ECM algorithm.
"""
@inline function support_change(R::AbstractMatrix{Float64}, old::AbstractMatrix{Float64})
  c = 0
  @inbounds for j in 2:size(R, 1), i in 1:j-1
    ((R[i, j] != 0) ⊻ (old[i, j] != 0)) && (c += 1)
  end
  return c
end

"""
    rel_log_change(λ, old)

Relative change of the scales on the log scale, `‖log λ − log old‖ / ‖log λ‖`.
Used in the stopping rule of the ECM algorithm.
"""
@inline function rel_log_change(λ::Vector{Float64}, old::Vector{Float64})
  num = 0.0
  den = 0.0
  @inbounds for j in eachindex(λ)
    lj = log(λ[j])
    num += (lj - log(old[j]))^2
    den += lj^2
  end
  return sqrt(num) / sqrt(max(den, eps()))
end

"""
    ecm(λ0, R0, η0, n, ρ, γ; ws = ADMMWorkspace(I - R0), solution_tol = 2.0^-12,
        R_update_tol = 2.0^-7, max_iter = ceil(Int, 2^4 * sqrt(p)), monitor_obj = true)

ECM algorithm at fixed `γ`.  Each iteration consists of the E-step and the η-,
R- and Λ-updates of the M-step, in that order.  The iterations stop when the
estimate and the log-posterior change by less than `solution_tol` (relative
change of `log λ`, number of pairs entering or leaving the support of `R`,
change of `η` and, if `monitor_obj`, relative change of the log-posterior), or
after `max_iter` iterations.  The ADMM workspace `ws` is supplied by the caller,
so that it is warm-started across the stages of the continuation path.  `ρ` must
be a symmetric `Matrix`.  Returns `(λ, R, η)`.
"""
function ecm(λ0::Vector{Float64}, R0::Matrix{Float64}, η0::Float64,
  n::Int, ρ::AbstractMatrix{Float64}, γ::Float64;
  ws::ADMMWorkspace=ADMMWorkspace(I - R0),
  solution_tol::Float64=2.0^-12,
  R_update_tol::Float64=2.0^-7,
  max_iter::Int=ceil(Int, 2^4 * sqrt(size(ρ, 1))),
  monitor_obj::Bool=true)
  p = size(ρ, 1)
  ℓ = 0.5 * p * (p - 1)                # number of pairs

  λ = copy(λ0)
  R = copy(R0)
  η = η0
  σ = Matrix{Float64}(undef, p, p)     # slab-membership probabilities
  wts = Matrix{Float64}(undef, p, p)   # weights γ(1 − σ)/2 of the R-update
  Σ̂ = Matrix{Float64}(undef, p, p)     # Λ ρ Λ
  ρR = Matrix{Float64}(undef, p, p)    # ρ ∘ R

  old_λ = similar(λ)
  old_R = similar(R)
  obj = monitor_obj ? log_posterior(λ, R, η, n, ρ, γ) : 0.0

  t = 0
  converged = false
  while !converged && t < max_iter
    t += 1

    # E-step: slab-membership probabilities and weights of the R-update
    slab_membership!(σ, R, η, γ)
    @inbounds @. wts = (1 - σ) * (γ / 2)

    # η-update: average slab-membership probability over the ℓ pairs
    old_η = η
    η = sum_upper(σ) / ℓ

    # R-update: weighted graphical lasso on Σ̂ = Λ ρ Λ
    copyto!(old_R, R)
    @inbounds Σ̂ .= λ .* ρ .* transpose(λ)     # (not @. : λ' would broadcast elementwise)
    update_R!(R, n, Σ̂, wts, ws, R_update_tol)

    # Λ-update: coordinate descent on ρ ∘ R
    copyto!(old_λ, λ)
    @inbounds @. ρR = ρ * R
    update_λ!(λ, n, ρR)

    # stopping rule
    converged = rel_log_change(λ, old_λ) < solution_tol &&
                support_change(R, old_R) < solution_tol * ℓ &&
                abs(η - old_η) < solution_tol
    if monitor_obj
      new_obj = log_posterior(λ, R, η, n, ρ, γ)
      converged &= abs(new_obj - obj) < solution_tol * max(1, abs(obj))
      obj = new_obj
    end
  end
  return λ, R, η
end

"""
    _ForestTable

Maximal configurations of the forest theorem for given `(m̄, d̄)`, used by
`γ_q`.  A configuration `(k₀, τ_1, K^b_1, K^s_1, τ_2, K^b_2, K^s_2)` is stored
with its sets of ranks as bitmasks (bit `k−1` ↔ rank `k`).  The factor `ν` of
each endpoint depends on the configuration only through `(τ, k₀, K^s)`, so each
distinct key is stored once, and the configurations point to it through `ι1`
and `ι2`.  The table depends only on `(m̄, d̄)` and is memoized.  Its size grows
quickly with `m̄` (at `d̄ = m̄`: 791 configurations for `m̄ = 5`, 3983 for
`m̄ = 6`, 19297 for `m̄ = 7`).
"""
struct _ForestTable
  k₀::Vector{Int}     # per configuration: rank of the central edge
  Kb1::Vector{Int}    # per configuration: bitmask of K^b_1
  Kb2::Vector{Int}    # per configuration: bitmask of K^b_2
  ι1::Vector{Int}     # per configuration: index of the ν key of endpoint 1
  ι2::Vector{Int}     # per configuration: index of the ν key of endpoint 2
  τν::Vector{Int}     # per ν key: τ
  k₀ν::Vector{Int}    # per ν key: k₀
  Ksν::Vector{Int}    # per ν key: bitmask of K^s
end

const _FOREST_TABLES = Dict{Tuple{Int,Int},_ForestTable}()   # memoized per (m̄, d̄)
const _FOREST_TABLES_LOCK = ReentrantLock()

"""
    _νf(x)

`ν(x) = x/(1 − e^{−x})`, with `ν(0) = 1`.
"""
_νf(x::Float64) = x > 0 ? x / -expm1(-x) : 1.0

"""
    _submasks(mask, k)

Subsets of `mask` with exactly `k` elements, as bitmasks.
"""
function _submasks(mask::Int, k::Int)
  out = Int[]
  s = mask
  while true
    count_ones(s) == k && push!(out, s)
    s == 0 && break
    s = (s - 1) & mask
  end
  return out
end

"""
    _forest_table(m̄, d̄)

Enumeration of the maximal configurations for `(m̄, d̄)`, keeping one of each
mirrored pair, as a `_ForestTable`.  The configurations are generated by their
shape and then by the assignment of the ranks `1, …, |c|` (see the extended
help of `γ_q`).
"""
function _forest_table(m̄::Int, d̄::Int)
  # Shape c = (db1, ds1, τ1, db2, ds2, τ2): numbers of edges of type b and s at
  # endpoints 1 and 2 (ds1 and ds2 count the central edge) and the indicators,
  # subject to the constraints that define a configuration.
  admissible(c) = c[2] ≥ 1 && c[5] ≥ 1 && c[3] ≤ min(c[1], 1) && c[6] ≤ min(c[4], 1) &&
                  c[1] + c[2] ≤ d̄ && c[4] + c[5] ≤ d̄ &&
                  c[1] + c[2] + c[4] + c[5] - 1 + c[3] + c[6] ≤ m̄

  k₀ = Int[]; Kb1 = Int[]; Kb2 = Int[]; ι1 = Int[]; ι2 = Int[]
  τν = Int[]; k₀ν = Int[]; Ksν = Int[]
  keyidx = Dict{NTuple{3,Int},Int}()
  function key!(τ, k, Ks)
    get!(keyidx, (τ, k, Ks)) do
      push!(τν, τ); push!(k₀ν, k); push!(Ksν, Ks)
      length(τν)
    end
  end

  for db1 in 0:d̄, ds1 in 1:d̄, τ1 in 0:1, db2 in 0:d̄, ds2 in 1:d̄, τ2 in 0:1
    c = (db1, ds1, τ1, db2, ds2, τ2)
    admissible(c) || continue
    any(t -> admissible(ntuple(s -> c[s] + (s == t), 6)), 1:6) && continue   # not maximal
    nc = db1 + ds1 + db2 + ds2 - 1                                          # |c|
    full = (1 << nc) - 1
    for k in 1:nc
      rest = full ⊻ (1 << (k - 1))
      for B1 in _submasks(rest, db1), S1 in _submasks(rest ⊻ B1, ds1 - 1),
          B2 in _submasks(rest ⊻ B1 ⊻ S1, db2)
        S2 = rest ⊻ B1 ⊻ S1 ⊻ B2
        isless((τ2, B2, S2), (τ1, B1, S1)) && continue   # keep one of each mirrored pair
        push!(k₀, k); push!(Kb1, B1); push!(Kb2, B2)
        push!(ι1, key!(τ1, k, S1)); push!(ι2, key!(τ2, k, S2))
      end
    end
  end
  return _ForestTable(k₀, Kb1, Kb2, ι1, ι2, τν, k₀ν, Ksν)
end

"""
    _forest_table_cached(m̄, d̄)

`_forest_table(m̄, d̄)`, memoized and thread-safe.
"""
function _forest_table_cached(m̄::Int, d̄::Int)
  lock(_FOREST_TABLES_LOCK) do
    get!(() -> _forest_table(m̄, d̄), _FOREST_TABLES, (m̄, d̄))
  end
end

"""
    _kw(uname, u, aname, a, default)

Value of a keyword given either under its Unicode name `uname` (value `u`) or
under its ASCII synonym `aname` (value `a`), or `default` when neither is given.
Throws an `ArgumentError` when both are given.
"""
function _kw(uname::Symbol, u, aname::Symbol, a, default)
  u === nothing && return a === nothing ? default : a
  a === nothing || throw(ArgumentError("give the keyword $uname or its ASCII synonym $aname, not both"))
  return u
end

"""
    γ_q(n, p; q = 0.05, m_max = 5, d_max = 1, nreps = 20_000, rng = default_rng())
    gamma_q(n, p; ...)

Calibrated hyperparameter `γ_q(m̄, d̄)`, with `m̄ = m_max` and `d̄ = d_max`, for
sample size `n` and `p` variables: the `(1 − q)`-quantile, under a diagonal
ground truth, of the limit law of the lower bound of the forest theorem,
computed by Monte Carlo with `nreps` replicates.  `gamma_q` is an ASCII alias,
and `m_max` and `d_max` can also be given under their Unicode names `m̄` and
`d̄`.

# Extended help

The forest theorem compares the diagonal model with the estimates supported on
forests with at most `m̄` edges and maximum degree at most `d̄`, through the
lower bound

    γ̲(m̄, d̄) = max{γ_R(m̄, d̄), γ_η(m̄)}.

By the joint order-statistics lemma, `{ℓ F̄_n(|ρ_n|_(k))}_{k ≤ m̄}` converges in
distribution to `(Γ_k)_{k ≤ m̄}`, with `Γ_k = E_1 + … + E_k`, the `E_i`
independent standard exponential, and `ℓ = p(p−1)/2`.  Substituting
`|ρ_n|_(k) = F̄_n^{-1}(Γ_k/ℓ) =: ϱ_k` and `w_(k) = 1/(1 − ϱ_k²)` in `γ̲(m̄, d̄)`
gives

    γ_R(m̄, d̄) = (n−1) max_c ϱ_{k₀} √(ω_1(c) ω_2(c)),
    ω_1(c)     = (1 + Σ_{k∈K^b_1} (w_(k) − 1)) · ν[(1+τ_1)/2 (log w_(k₀) + Σ_{k∈K^s_1} log w_(k))],
    γ_η(m̄)    = 1 + (1/ℓ) Σ_{k=1}^{m̄} w_(k)^{(n−1)/2},

with `ν(x) = x/(1 − e^{−x})`, `ω_2(c)` defined by exchanging the endpoints 1
and 2 of the central edge, and the maximum running over the configurations
`c = (k₀, K^b_1, K^s_1, τ_1, K^b_2, K^s_2, τ_2)` of the paper.  Every rank is
at most `|c| ≤ m̄`, so `m̄` draws `Γ_1, …, Γ_m̄` per replicate suffice.
`γ_q(m̄, d̄)` depends on the data only through `n` and `p`.

`F̄_n` is inverted through the Student form of Theorem 5.1.3 of Muirhead (1982),

    F̄_n^{-1}(v) = z_v / √(n − 2 + z_v²),   z_v = (1 − v/2)-quantile of t_{n−2},

so `1 − ϱ_k² = (n−2)/(n−2 + z_v²)` is evaluated directly, without cancellation
when `ϱ_k ≈ 1`, which is where `γ_η` is largest.  As in the paper, `ϱ_k := 0`
when `Γ_k > ℓ`, where `F̄_n^{-1}` is undefined, so that `w_(k) = 1`.

Two reductions keep the maximum exact and cheap.

  * Only maximal configurations are enumerated, since the maximum is attained
    at one (lemma on maximal configurations).  The configurations are generated
    by their shape, the numbers of edges of each type at each endpoint and the
    two indicators, and then by the assignment of the ranks `1, …, |c|`.  A
    configuration is maximal exactly when its shape admits no further edge
    (with the new rank `|c| + 1`) and no further indicator.
  * Exchanging the endpoints 1 and 2 leaves `ϱ_{k₀} √(ω_1 ω_2)` unchanged, so
    only one configuration of each mirrored pair is kept.

Per replicate, `Σ(w − 1)` and `Σ log w` are tabulated over all `2^m̄` bitmasks,
`ν` is evaluated once per distinct `(τ, k₀, K^s)`, and each configuration costs
four lookups and five multiplications.
"""
function γ_q(n::Int, p::Int; q::Float64=0.05, m̄=nothing, m_max=nothing,
  d̄=nothing, d_max=nothing, nreps::Int=20_000, rng::AbstractRNG=default_rng())

  # ASCII synonyms of the Unicode keywords
  m̄ = Int(_kw(:m̄, m̄, :m_max, m_max, 5))
  d̄ = Int(_kw(:d̄, d̄, :d_max, d_max, 1))

  n ≥ 3 || throw(ArgumentError("γ_q needs n ≥ 3 (F̄_n has n − 2 degrees of freedom); got n = $n"))
  p ≥ 2 || throw(ArgumentError("γ_q needs p ≥ 2; got p = $p"))
  0 < q < 1 || throw(ArgumentError("γ_q needs q ∈ (0, 1); got q = $q"))
  1 ≤ m̄ ≤ p - 1 || throw(ArgumentError("γ_q needs 1 ≤ m_max ≤ p − 1; got m_max = $m̄"))
  1 ≤ d̄ ≤ m̄ || throw(ArgumentError("γ_q needs 1 ≤ d_max ≤ m_max; got d_max = $d̄"))

  tab = _forest_table_cached(m̄, d̄)
  nkeys = length(tab.τν)
  nconf = length(tab.k₀)

  df = n - 2                          # degrees of freedom of the Student form
  ℓ = p * (p - 1) / 2
  half = (n - 1) / 2
  u = Vector{Float64}(undef, m̄)      # u_k = 1 − ϱ_k² = 1/w_(k)
  r² = Vector{Float64}(undef, m̄)     # ϱ_k²
  bw = Vector{Float64}(undef, m̄)     # w_(k) − 1
  lw = Vector{Float64}(undef, m̄)     # log w_(k)
  sum_bw = zeros(1 << m̄)             # sum_bw[K+1] = Σ_{k∈K} (w_(k) − 1)
  sum_lw = zeros(1 << m̄)             # sum_lw[K+1] = Σ_{k∈K} log w_(k)
  νv = Vector{Float64}(undef, nkeys)
  g = Vector{Float64}(undef, nreps)

  @inbounds for b in 1:nreps
    Γ = 0.0
    for k in 1:m̄
      Γ += randexp(rng)                       # Γ_k = E_1 + … + E_k
      v = Γ / ℓ
      if v < 1
        z = tdistinvcdf(df, 1 - v / 2)
        den = df + z * z
        u[k] = df / den
        r²[k] = z * z / den
      else
        u[k] = 1.0                            # ϱ_k := 0
        r²[k] = 0.0
      end
      bw[k] = r²[k] / u[k]
      lw[k] = -log(u[k])
    end

    # subset sums over all bitmasks, each from the mask without its lowest bit
    for K in 1:(1 << m̄) - 1
      k = trailing_zeros(K) + 1
      K′ = K & (K - 1)
      sum_bw[K + 1] = sum_bw[K′ + 1] + bw[k]
      sum_lw[K + 1] = sum_lw[K′ + 1] + lw[k]
    end

    for t in 1:nkeys
      νv[t] = _νf((1 + tab.τν[t]) / 2 * (lw[tab.k₀ν[t]] + sum_lw[tab.Ksν[t] + 1]))
    end

    best = 0.0                                # max over configurations of ϱ_{k₀}² ω_1 ω_2
    for l in 1:nconf
      cand = r²[tab.k₀[l]] * (1 + sum_bw[tab.Kb1[l] + 1]) * νv[tab.ι1[l]] *
             (1 + sum_bw[tab.Kb2[l] + 1]) * νv[tab.ι2[l]]
      best = max(best, cand)
    end
    γ_R = (n - 1) * sqrt(best)

    s = 0.0
    for k in 1:m̄
      s += u[k]^(-half)
    end
    γ_η = 1 + s / ℓ

    g[b] = max(γ_R, γ_η)
  end

  return quantile!(g, 1 - q)                  # quantile! sorts g in place
end

const gamma_q = γ_q          # ASCII alias

"""
    _pcss_fit_usage()

Throws the `ArgumentError` of `pcss_fit` called without a usable input, with a
usage message.
"""
function _pcss_fit_usage()
  throw(ArgumentError("""
    pcss_fit expects either a data matrix or a (sample size, covariance) pair,
    passed as keyword arguments:
        pcss_fit(X = data)                     # X is the n × p data matrix
        pcss_fit(n = size(X, 1), S = cov(X))   # S is the p × p sample covariance
    Optional keywords: gamma, criterion, q, m_max, d_max, nreps, rng, gamma_ebic,
    n_path, gamma0, Omega0, diag_cmp, gamma_grid, K, span (the Unicode names
    γ, m̄, d̄, γ_ebic, γ0, Ω0, γ_grid are also accepted).
    If both X and (n, S) are given, X takes precedence."""))
end

"""
    pcss_fit(; X, gamma = nothing, criterion = :gamma_q, q = 0.05, m_max = 5, d_max = 1, ...)
    pcss_fit(; n, S, ...)

PCSS estimate of the scales `lambda` (the diagonal of Λ), the partial correlation
matrix `R` and the mixture weight `eta`, from the `n × p` data matrix `X` or from
the sample size `n` and the `p × p` sample covariance `S`.  Returns
`(lambda, R, eta, gamma)`, and throws an `ArgumentError` on a usage error.
The edges of the estimated graph are `pcss_edges(fit)`.

The hyperparameter γ is `gamma` when given.  Otherwise it is set by `criterion`:
  * `:gamma_q` (default, also accepted as `:γ_q`): the calibrated value
    `gamma_q(n, p; q, m_max, d_max, nreps, rng)`;
  * `:ebic`: the minimizer of the EBIC, with parameter `gamma_ebic` (0.5 by
    default), over a grid;
  * `:bic`: the minimizer of the BIC over the same grid.
The grid is `gamma_grid` when given, and otherwise has `K` values, log-spaced
from `γ_q/span` to `γ_q * span`.

The estimate is the output of the procedure of the paper: ECM fits along a
continuation path of `n_path` values of γ, log-spaced from `gamma0`
(`(n − 1)/√n` by default) to the target γ, followed, if `diag_cmp`, by a
comparison with the diagonal estimate.  If `gamma0` exceeds the target, or
`n_path == 1`, the path is replaced by a single fit at the target.  `Omega0`
replaces the diagonal estimate as the start of the path, on the correlation
scale, with η set to the fraction of nonzero off-diagonal entries of `Omega0`.
A dense `Omega0` therefore gives η = 1, which leaves no weight on the spike: the
R-update is then unpenalized at every stage, and for n < p it has no solution.
`Omega0` should be sparse.

The keywords `gamma`, `m_max`, `d_max`, `gamma_ebic`, `gamma0`, `Omega0` and
`gamma_grid` can also be given under their Unicode names `γ`, `m̄`, `d̄`,
`γ_ebic`, `γ0`, `Ω0` and `γ_grid`.  Giving both names of one keyword throws an
`ArgumentError`.
"""
function pcss_fit(; X=nothing, n=nothing, S=nothing, γ=nothing, gamma=nothing,
  criterion::Symbol=:gamma_q, q::Float64=0.05, m̄=nothing, m_max=nothing,
  d̄=nothing, d_max=nothing, nreps::Int=20_000, rng::AbstractRNG=default_rng(),
  γ_ebic=nothing, gamma_ebic=nothing, n_path::Int=4, γ0=nothing, gamma0=nothing,
  Ω0=nothing, Omega0=nothing, diag_cmp::Bool=true, γ_grid=nothing, gamma_grid=nothing,
  K::Int=25, span::Float64=8.0)

  # ASCII synonyms of the Unicode keywords
  γ = _kw(:γ, γ, :gamma, gamma, nothing)
  m̄ = Int(_kw(:m̄, m̄, :m_max, m_max, 5))
  d̄ = Int(_kw(:d̄, d̄, :d_max, d_max, 1))
  γ_ebic = Float64(_kw(:γ_ebic, γ_ebic, :gamma_ebic, gamma_ebic, 0.5))
  γ0 = _kw(:γ0, γ0, :gamma0, gamma0, nothing)
  Ω0 = _kw(:Ω0, Ω0, :Omega0, Omega0, nothing)
  γ_grid = _kw(:γ_grid, γ_grid, :gamma_grid, gamma_grid, nothing)

  # n, p, the sample correlation matrix ρ and the sample standard deviations sd,
  # from X, or from (n, S); X takes precedence.  ρ is stored as a plain symmetric
  # Matrix, and sd_i = √(S_n)_ii returns λ to the data scale at the end.
  if X isa AbstractMatrix
    n, p = size(X)
    ρ = Matrix(Symmetric(cor(X)))
    sd = std.(eachcol(X), corrected=false)
  elseif n isa Integer && n ≥ 1 && S isa AbstractMatrix && size(S, 1) == size(S, 2)
    n = Int(n)
    p = size(S, 1)
    sd = sqrt.(diag(S))
    C = Matrix{Float64}(S) ./ (sd * sd')
    @inbounds for i in 1:p
      C[i, i] = 1.0                                # exact unit diagonal
    end
    ρ = Matrix(Symmetric(C))
  else
    _pcss_fit_usage()
  end

  criterion === :γ_q && (criterion = :gamma_q)       # the Unicode spelling is accepted too
  criterion in (:gamma_q, :bic, :ebic) ||
    throw(ArgumentError("pcss_fit: criterion must be :gamma_q, :bic or :ebic (got :$criterion)"))

  # selection by EBIC or BIC over a grid, each value of γ fitted independently
  if γ === nothing && criterion !== :gamma_q
    sel = _ic_select(n, p, ρ, sd; criterion=criterion, γ_ebic=γ_ebic,
      q=q, m̄=m̄, d̄=d̄, nreps=nreps, rng=rng,
      n_path=n_path, γ_grid=γ_grid, K=K, span=span, γ0=γ0, diag_cmp=diag_cmp)
    return (lambda=sel.lambda, R=sel.R, eta=sel.eta, gamma=sel.gamma)
  end

  γend::Float64 = isnothing(γ) ? γ_q(n, p; q=q, m̄=m̄, d̄=d̄, nreps=nreps, rng=rng) : Float64(γ)
  # Default start of the path, γ0 = (n − 1)/√n: at the diagonal estimate, R = 0
  # satisfies the optimality condition of the first R-update for the pair (i, j)
  # exactly when |ρ_ij| ≤ 1/√n.
  γini::Float64 = isnothing(γ0) ? (n - 1) / sqrt(n) : Float64(γ0)

  # Start of the path: the diagonal estimate on the correlation scale
  # (λ_i = √(1 − 1/n), R = 0, η = 0), or the decomposition of Ω0 into scales and
  # partial correlations, with η the fraction of nonzero off-diagonal entries of
  # its R.  From the diagonal estimate, η = 0 is a fixed point of the ECM
  # algorithm, so η stays 0 until it is reset below.
  if Ω0 !== nothing
    Ω = Matrix{Float64}(Ω0)
    d = sqrt.(diag(Ω))                            # λ_i = √Ω_ii
    λ = copy(d)
    R = -Ω ./ (d * d')                            # r_ij = −Ω_ij/√(Ω_ii Ω_jj)
    R[diagind(R)] .= 0.0
    η = count(!iszero, R) / (p * (p - 1))
  else
    λ = sqrt(1 - 1 / n) .* ones(p)
    R = zeros(p, p)
    η = 0.0
  end

  # one ADMM workspace, warm-started across all the stages of the path
  ws = ADMMWorkspace(I - R)

  diag_obj = diag_cmp ? log_posterior(sqrt(1 - 1 / n) .* ones(p), zeros(p, p), 0.0, n, ρ, γend) : 0.0

  # Continuation path: n_path values of γ, log-spaced from γini to γend.  Before
  # each stage after the first, η is reset to the fraction of nonzero
  # off-diagonal entries of R, since it would otherwise remain zero.
  # Only the estimate at γend is kept.
  γpath = (n_path == 1 || γini > γend) ? [γend] : exp.(LinRange(log(γini), log(γend), n_path))
  for (i, γi) in enumerate(γpath)
    i > 1 && (η = count(!iszero, R) / (p * (p - 1)))
    λ, R, η = ecm(λ, R, η, n, ρ, γi; ws=ws)
  end

  # Comparison with the diagonal estimate: return whichever has the larger
  # log-posterior at γend.  A non-finite log-posterior, which signals a
  # numerically non-positive-definite I − R, also returns the diagonal estimate.
  if diag_cmp
    obj_fit = log_posterior(λ, R, η, n, ρ, γend)
    if !isfinite(obj_fit) || obj_fit < diag_obj
      λ = sqrt(1 - 1 / n) .* ones(p)
      R = zeros(p, p)
      η = 0.0
    end
  end

  # back to the data scale
  λ ./= sd
  return (lambda=λ, R=R, eta=η, gamma=γend)
end

"""
    pcss_edges(fit)

Edges of the graph estimated by `fit`, the output of `pcss_fit`: the pairs
`(i, j)`, `i < j`, for which the slab is more probable than the spike,

    |r_ij| > r_c(η) = (1/γ)[log γ̃ + log((1 − η)/η)],   γ̃ = γ/(1 − e^{−γ}),

at the estimated `R` and `η` and at the `γ` of the fit.  Returns a vector of
pairs `(i, j)` in lexicographic order, which is empty for the diagonal estimate
(`eta = 0`).  `fit` can be any object with fields `R`, `eta` and `gamma`, such
as the named tuple `(R = R, eta = eta, gamma = gamma)`.
"""
function pcss_edges(fit)
  R, η, γ = fit.R, fit.eta, fit.gamma
  η > 0 || return Tuple{Int,Int}[]               # diagonal estimate: r_c = ∞
  r_c = (log(γ / -expm1(-γ)) + log((1 - η) / η)) / γ
  p = size(R, 1)
  return [(i, j) for i in 1:p for j in i+1:p if abs(R[i, j]) > r_c]
end

"""
    ic_score(λ, R, n, ρ; criterion = :ebic, γ_ebic = 0.5)

EBIC of a fit on the correlation scale, as in the paper,

    EBIC = −n[log det Ω − tr(ρ Ω)] + k log n + 4 γ_ebic k log p,

with `Ω = Λ(I − R)Λ` and `k` the number of nonzero entries of `R` above the
diagonal.  With `criterion = :bic` the last term is dropped.  Using `ρ` and the
standardized scale instead of `S_n` and `Λ` adds the same constant for every
`γ`, so the selected `γ` is the same.  `η` enters only the prior, and does not
count in `k`.  A non-positive-definite `I − R` gives `+Inf`, so that the
corresponding `γ` is never selected.
"""
function ic_score(λ::Vector{Float64}, R::AbstractMatrix{Float64},
  n::Int, ρ::AbstractMatrix{Float64}; criterion::Symbol=:ebic, γ_ebic::Float64=0.5)
  p = length(λ)
  ld, quad = logdet_and_quad(λ, R, ρ)
  isfinite(ld) || return Inf
  loglik = n * (sum(log, λ) + 0.5 * (ld - quad))         # (n/2)[log det Ω − tr(ρΩ)]
  k = 0
  @inbounds for j in 2:p, i in 1:j-1
    R[i, j] != 0 && (k += 1)
  end
  ic = -2loglik + k * log(n)
  criterion === :ebic && (ic += 4 * γ_ebic * k * log(p))
  return ic
end

"""
    _ic_select(n, p, ρ, sd; criterion, γ_ebic, q, m̄, d̄, nreps, rng, n_path,
               γ_grid, K, span, γ0, diag_cmp)

Selection of `γ` by EBIC or BIC over a grid (the `:ebic` and `:bic` criteria of
`pcss_fit`).  Each value of the grid is fitted independently, with its own
continuation path started from the diagonal estimate, so that its estimate does
not depend on those at the other values.  The default grid has `K` values,
log-spaced from `γ_q/span` to `γ_q * span`.  The diagonal estimate is also a
candidate, returned with `gamma = Inf` if its criterion is the smallest.
Returns `(lambda, R, eta, gamma, ic, grid, ics)`, with `lambda` on the data
scale.
"""
function _ic_select(n::Int, p::Int, ρ::AbstractMatrix{Float64}, sd::Vector{Float64};
  criterion::Symbol, γ_ebic::Float64, q::Float64, m̄::Int, d̄::Int, nreps::Int,
  rng::AbstractRNG, n_path::Int, γ_grid, K::Int, span::Float64, γ0, diag_cmp::Bool)

  grid = if γ_grid !== nothing
    sort!(collect(Float64, γ_grid))
  else
    γ_q(n, p; q=q, m̄=m̄, d̄=d̄, nreps=nreps, rng=rng) .* exp.(LinRange(-log(span), log(span), K))
  end

  bestλ = sqrt(1 - 1 / n) .* ones(p)             # diagonal estimate, correlation scale
  bestR = zeros(p, p)
  bestη = 0.0
  bestγ = Inf
  bestic = ic_score(bestλ, bestR, n, ρ; criterion=criterion, γ_ebic=γ_ebic)
  ics = Vector{Float64}(undef, length(grid))

  for (t, γ) in enumerate(grid)
    # Fit at the given γ.  S = ρ has unit diagonal, so sd ≡ 1 inside and lambda
    # is returned on the correlation scale.
    fit = pcss_fit(; n=n, S=ρ, γ=γ, n_path=n_path, γ0=γ0, diag_cmp=diag_cmp)
    b = ic_score(fit.lambda, fit.R, n, ρ; criterion=criterion, γ_ebic=γ_ebic)
    ics[t] = b
    if b < bestic
      bestic = b
      bestλ = fit.lambda
      bestR = fit.R
      bestη = fit.eta
      bestγ = fit.gamma
    end
  end

  bestλ = bestλ ./ sd                            # correlation scale → data scale
  return (lambda=bestλ, R=bestR, eta=bestη, gamma=bestγ, ic=bestic, grid=grid, ics=ics)
end

end # module
