using PCSS, Test, Random, LinearAlgebra, Statistics

@testset "PCSS" begin
  @testset "γ_q" begin
    g1 = γ_q(64, 32; nreps = 2_000, rng = Xoshiro(0))
    @test g1 == γ_q(64, 32; nreps = 2_000, rng = Xoshiro(0))
    @test g1 > 0
    @test_throws ArgumentError γ_q(2, 32)
    @test gamma_q === γ_q
    g2 = γ_q(64, 32; m̄ = 4, d̄ = 3, nreps = 2_000, rng = Xoshiro(0))
    @test g2 == γ_q(64, 32; m_max = 4, d_max = 3, nreps = 2_000, rng = Xoshiro(0))
    @test_throws ArgumentError γ_q(64, 32; m̄ = 4, m_max = 4)
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
end
