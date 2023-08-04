using Test

@testset "var" begin
    σ²s = range(0, 1, 10000)[2:end-1]
    for σ² in σ²s
        κ = var2κ(σ²)
        d = VonMises(κ)
        @test isapprox(σ², var(d), atol = 1e-5)
    end
end

@testset "std" begin
    σs = range(0, sqrt(2), 10000)[2:end-1]
    for σ in σs
        κ = std2κ(σ)
        d = VonMises(κ)
        @test isapprox(σ, std(d), atol = 1e-5)
    end
end
