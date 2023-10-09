# using Statistics, 
using LinearAlgebra
using DistributionsAD, StaticArrays
using GLMakie, Folds
using Turing

# functions:
function next_step(θ, brwθ, crwθ, w)
    brwyx = sincos(brwθ)
    crwyx = sincos(θ + crwθ)
    y, x = @. w*brwyx + (1 - w)*crwyx
    return (atan(y, x), SVector{2, Float64}(x, y))
end

function get_exit_azimuth(nsteps, brwκ, crwκ, w)
    brw = VonMises(brwκ)
    crw = VonMises(crwκ)
    xy = zero(SVector{2, Float64})
    θ = 0.0
    while norm(xy) < nsteps
        θ, Δ = next_step(θ, rand(brw), rand(crw), w)
        xy += Δ
    end
    return atan(reverse(xy)...)
end

DataFrames.transform!(df, :degrees => ByRow(x -> deg2rad.(acosd.(cosd.(x)))) => :radians)
DataFrames.transform!(df, :radians => ByRow(x -> atan(mean(sin.(x)), mean(cos.(x)))) => :μ)
DataFrames.transform!(df, [:radians, :μ] => ByRow((x, μ) -> x .- μ) => :centered)



brwκ = 15
crwκ = 4
w = 0.5
nsteps = 20
α = Folds.map(_ -> get_exit_azimuth(nsteps, brwκ, crwκ, w), 1:10^5)
d = fit(Normal, α)
pdf.(d, vcat(df.centered...))



fig = Figure()
ax = Axis(fig[1,1])
hist!(ax, α, bins = 100, normalization=:pdf)
sg = SliderGrid(fig[2, 1],
                (label = "κ", range = 0.001:0.001:0.2)
)
κ = sg.sliders[1].value
x = range(extrema(α)..., 101)
lines!(ax, x, @lift(pdf.(Normal(0, $κ), x)))
on(κ) do _
    autolimits!(ax)
end

x = -pi:0.01:pi
p = VonMises(1)
lines(x, pdf.(p, x))

n = 100_000
α = rand(p, n);

θ = 2
findfirst(>(θ), sort(α))/n
cdf(p, θ)

ps = range(0, 1, n)
lines(diff(sort(α)))

lines(sort(α), ps)
ps = range(-pi, pi, n)
lines!(ps, cdf.(p, ps))


nsteps = 18
@model function one_exit(α)
    brwκ ~ Uniform(0.01, 400)
    crwκ ~ Truncated(Normal(4, 1), 0.001, 400)
    w ~ Uniform(0, 1)
    α = get_exit_azimuth(nsteps, brwκ, crwκ, w)
end
α = rand(VonMises(2))
m = one_exit(α)
chain = sample(m, NUTS(), 1_000);
