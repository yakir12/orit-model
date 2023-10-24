# using Statistics, 
using LinearAlgebra
using DistributionsAD, StaticArrays
using GLMakie, Folds
using Turing



@model function beetle_exits(r)
    nsteps = 20
    crw = VonMises(4)
    brw_κ ~ Uniform(0.01, 400)
    w ~ Uniform(0, 1)
end




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

function get_exit_point(nsteps, brw, crw, w)
    xy = zero(SVector{2, Float64})
    θ = 0.0
    while norm(xy) < nsteps
        θ, Δ = next_step(θ, rand(brw), rand(crw), w)
        xy += Δ
    end
    return nsteps*normalize(xy) # shorten / extend the last step of the beetle so it crosses the nsteps circle around the origin
end

mean_resultant_length(nrepetitions, nsteps, brw, crw, w) = norm(mapreduce(_ -> get_exit_point(nsteps, brw, crw, w), +, 1:nrepetitions)/nrepetitions/nsteps) 

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


function probability(nsteps, brw_κ, crw, w)
    n = 1000
    brw = VonMises(brw_κ)
    rvalues = Folds.map(i -> mean_resultant_length(10, nsteps, brw, crw, w), 1:n)
    return fit(Beta, rvalues)
end

@model function beetle_exits(r)
    nsteps = 20
    crw = VonMises(4)
    brw_κ ~ Uniform(0.01, 400)
    w ~ Uniform(0, 1)
    r ~ sum([rand(VonMises(brw_κ)) for _ in 1:10])
end

function generate()
    nsteps = 20
    crw = VonMises(4)
    brw_κ = 6
    w = 0.3
    d = probability(nsteps, brw_κ, crw, w)
    r = mode(d)
end

iterations = 100
ϵ = 0.05
τ = 10
data = [generate() for _ in 1:100]


# Start sampling.
chain = sample(beetle_exits(data), HMC(ϵ, τ), iterations)



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

