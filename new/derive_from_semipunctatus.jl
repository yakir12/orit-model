using Statistics, LinearAlgebra
using Distributions, StaticArrays
# using DataFrames, AlgebraOfGraphics
# using GLMakie
using ProgressMeter

# functions:
function next_step(θ, brwθ, crwθ, w)
    brwyx = sincos(brwθ)
    crwyx = sincos(θ + crwθ)
    y, x = @. w*brwyx + (1 - w)*crwyx
    return (atan(y, x), SVector{2, Float64}(x, y))
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

### main

n = 10
nd = 10000
crw_κ = 4 # equivalent to an "angular deviation" of 30°
crw = VonMises(crw_κ)
w = range(0, 1, n)
brw_κ = exp.(range(log(0.01), log(400), n))
brw = VonMises.(brw_κ)
nsteps = range(10, 40, n)

nrepetitions = 10
dists = Array{Float64}(undef, n, n, n, nd);
p = Progress(length(dists))
Threads.@threads for ijkl in eachindex(IndexCartesian(), dists)
    i, j, k, _ = Tuple(ijkl)
    dists[ijkl] = mean_resultant_length(nrepetitions, nsteps[i], brw[j], crw, w[k])
    next!(p)
end
finish!(p)


nrepetitions = 100_000
μ = Array{Float64}(undef, n, n, n);
p = Progress(length(μ))
Threads.@threads for ijk in eachindex(IndexCartesian(), μ)
    i, j, k = Tuple(ijk)
    μ[ijk] = mean_resultant_length(nrepetitions, nsteps[i], brw[j], crw, w[k])
    next!(p)
end
finish!(p)

using GLMakie

i, j, k = (n,n,n) .- 5
data = dists[i, j, k, :]
d = fit(Beta, data)
x = 0:0.01:1
y = pdf.(d, x)
hist(data, bins=25, normalization=:pdf)
lines!(x, y)
vlines!(μ[i, j, k])

r = [[mean_resultant_length(nrepetitions, nsteps[5], brw[5], crw, w[5]) for _ in 1:1000] for nrepetitions in (10, 1000)]

fig = Figure()
for i in 1:2
    ax = Axis(fig[i,1], limits=((0.9,1), nothing))
    hist!(ax, r[2], bins=25)
end

using Folds
mean_resultant_length(nrepetitions, nsteps, brw, crw, w) = norm(Folds.mapreduce(_ -> get_exit_point(nsteps, brw, crw, w), +, 1:nrepetitions)/nrepetitions/nsteps) 

nrepetitionss = 10 .^ (1:9)
r = zeros(length(nrepetitionss))
p = Progress(length(nrepetitionss))
Threads.@threads for i in eachindex(nrepetitionss)
    r[i] = mean_resultant_length(nrepetitionss[i], nsteps[5], brw[5], crw, w[5])
    next!(p)
end
finish!(p)
scatterlines(diff(nrepetitionss), abs.(diff(r)), axis=(;xscale=log10, yscale=log10))


nrepetitions = 100
r = [mean_resultant_length(nrepetitions, nsteps[5], brw[5], crw, w[5]) for _ in 1:100000]


