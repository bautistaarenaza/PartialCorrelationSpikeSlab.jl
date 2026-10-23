using PCSS, Test, Random, LinearAlgebra, Statistics

@testset "PCSS" begin
  @testset "γ_q" begin
    g1 = γ_q(64, 32; nreps = 2_000, rng = Xoshiro(0))
    @test g1 == γ_q(64, 32; nreps = 2_000, rng = Xoshiro(0))
    @test g1 > 0
    @test_throws ArgumentError γ_q(2, 32)
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
  end
end
