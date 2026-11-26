using PartialCorrelationSpikeSlab, Test, Random, LinearAlgebra, Statistics

@testset "PartialCorrelationSpikeSlab" begin
  @testset "γ_q" begin
    g1 = γ_q(64, 32; nreps = 2_000, rng = Xoshiro(0))
    @test g1 == γ_q(64, 32; nreps = 2_000, rng = Xoshiro(0))
    @test g1 > 0
    @test_throws ArgumentError γ_q(2, 32)
    @test gamma_q === γ_q
    g2 = γ_q(64, 32; m̄ = 4, d̄ = 3, nreps = 2_000, rng = Xoshiro(0))
    @test g2 == γ_q(64, 32; m_max = 4, d_max = 3, nreps = 2_000, rng = Xoshiro(0))
    @test_throws ArgumentError γ_q(64, 32; m̄ = 4, m_max = 4)
    # default m_max = min(5, p − 1)
    @test γ_q(64, 4; nreps = 500, rng = Xoshiro(0)) == γ_q(64, 4; m_max = 3, nreps = 500, rng = Xoshiro(0))
    @test_throws ArgumentError γ_q(64, 4; m_max = 5)
    @test_throws ArgumentError γ_q(64, 32; nreps = 0)
  end
  @testset "pcss_fit" begin
    n, p = 400, 10
    Ω = Matrix(SymTridiagonal(ones(p), fill(-0.4, p - 1)))
    X = randn(Xoshiro(1), n, p) * inv(cholesky(Ω).U)'
    γ = γ_q(n, p; nreps = 2_000, rng = Xoshiro(0))
    a = pcss_fit(; X = X, γ = γ)
    b = pcss_fit(; n = n, S = cov(X; corrected = false), γ = γ)
    @test a.R ≈ a.R' && all(iszero, diag(a.R))
    @test 0 ≤ a.eta ≤ 1
    @test a.R ≈ b.R atol = 1e-6
    @test a.lambda ≈ b.lambda rtol = 1e-6
    @test_throws ArgumentError pcss_fit()
    @test_throws ArgumentError pcss_fit(; n = 10, S = ones(3, 4))
    @test_throws ArgumentError pcss_fit(; X = X, criterion = :foo)
    c = pcss_fit(; X = X, criterion = :gamma_q, nreps = 2_000, rng = Xoshiro(0))
    d = pcss_fit(; X = X, criterion = :γ_q, nreps = 2_000, rng = Xoshiro(0))
    @test c.gamma == d.gamma == γ
    @test c.R == d.R
  end
  @testset "input checks" begin
    n, p = 50, 6
    X = randn(Xoshiro(3), n, p)
    S = cov(X; corrected = false)
    γ = 10.0
    Xc = copy(X); Xc[:, 2] .= 1.0                            # constant column
    @test_throws ArgumentError pcss_fit(; X = Xc, γ = γ)
    Xn = copy(X); Xn[3, 4] = NaN
    @test_throws ArgumentError pcss_fit(; X = Xn, γ = γ)
    Xm = Matrix{Union{Missing,Float64}}(X); Xm[1, 1] = missing
    @test_throws ArgumentError pcss_fit(; X = Xm, γ = γ)
    @test_throws ArgumentError pcss_fit(; X = X[:, 1:1], γ = γ)
    @test_throws ArgumentError pcss_fit(; X = X[1:1, :], γ = γ)
    Sa = copy(S); Sa[1, 2] += 1                              # not symmetric
    @test_throws ArgumentError pcss_fit(; n = n, S = Sa, γ = γ)
    Sd = copy(S); Sd[2, 2] = 0                               # zero variance
    @test_throws ArgumentError pcss_fit(; n = n, S = Sd, γ = γ)
    Sp = [1.0 0.9 0.9; 0.9 1.0 -0.9; 0.9 -0.9 1.0]           # not positive semidefinite
    @test_throws ArgumentError pcss_fit(; n = n, S = Sp, γ = γ)
    @test_throws ArgumentError pcss_fit(; X = X, γ = -1.0)
    @test_throws ArgumentError pcss_fit(; X = X, γ = γ, γ0 = 0.0)
    @test_throws ArgumentError pcss_fit(; X = X, γ = γ, n_path = 0)
    @test_throws ArgumentError pcss_fit(; X = X, γ = γ, Ω0 = ones(p, p))
    @test_throws ArgumentError pcss_fit(; X = X, γ = γ, Ω0 = Matrix(1.0I, p + 1, p + 1))
    @test_throws ArgumentError pcss_fit(; X = X, criterion = :ebic, γ_grid = Float64[])
    # default m_max = min(5, p − 1) for small p
    X4 = randn(Xoshiro(4), 100, 4)
    fit = pcss_fit(; X = X4, nreps = 500, rng = Xoshiro(0))
    @test fit.gamma == γ_q(100, 4; m_max = 3, nreps = 500, rng = Xoshiro(0))
  end
  @testset "ASCII keyword synonyms" begin
    n, p = 200, 8
    Ω = Matrix(SymTridiagonal(ones(p), fill(-0.4, p - 1)))
    X = randn(Xoshiro(2), n, p) * inv(cholesky(Ω).U)'
    γ = γ_q(n, p; nreps = 2_000, rng = Xoshiro(0))
    Ω0 = Matrix(1.0I, p, p)
    u = pcss_fit(; X = X, γ = γ, γ0 = 2γ, Ω0 = Ω0)
    a = pcss_fit(; X = X, gamma = γ, gamma0 = 2γ, Omega0 = Ω0)
    @test u == a
    grid = γ .* [0.5, 1.0, 2.0]
    u = pcss_fit(; X = X, criterion = :ebic, γ_ebic = 0.25, γ_grid = grid)
    a = pcss_fit(; X = X, criterion = :ebic, gamma_ebic = 0.25, gamma_grid = grid)
    @test u == a
    u = pcss_fit(; X = X, m̄ = 4, d̄ = 3, nreps = 2_000, rng = Xoshiro(0))
    a = pcss_fit(; X = X, m_max = 4, d_max = 3, nreps = 2_000, rng = Xoshiro(0))
    @test u == a
    @test_throws ArgumentError pcss_fit(; X = X, γ = γ, gamma = γ)
    @test_throws ArgumentError pcss_fit(; X = X, Ω0 = Ω0, Omega0 = Ω0)
  end
  @testset "pcss_edges" begin
    n, p = 400, 20
    Ω = Matrix(SymTridiagonal(ones(p), fill(-0.4, p - 1)))
    X = randn(Xoshiro(1), n, p) / cholesky(Ω).U'
    fit = pcss_fit(; X = X, rng = Xoshiro(0))
    e = pcss_edges(fit)
    @test e == [(i, i + 1) for i in 1:p-1]                # recovers the chain
    gt = fit.gamma / -expm1(-fit.gamma)
    r_c = (log(gt) + log((1 - fit.eta) / fit.eta)) / fit.gamma
    @test e == [(i, j) for i in 1:p for j in i+1:p if abs(fit.R[i, j]) > r_c]
    @test pcss_edges((R = fit.R, eta = fit.eta, gamma = fit.gamma)) == e
    @test isempty(pcss_edges((R = zeros(3, 3), eta = 0.0, gamma = Inf)))   # diagonal estimate
  end
end
