using LinearAlgebra
using Distributions, StaticArrays, Folds
using MonteCarloMeasurements
const SV = SVector{2, Particles{Float64, 2000}}

function next_step(θ, brw, crw, w)
    brwyx = sincos(brw)
    crwyx = sincos(θ + crw)
    y, x = @. w*brwyx + (1 - w)*crwyx
    return (atan(y, x), SV(x, y))
end
function get_exit_point(nsteps, brw, crw, w)
    xy = SV(Particles(), Particles())
    θ = Particles()
    for i in 1:nsteps
        θ, Δ = next_step(θ, brw, crw, w)
        xy += Δ
    end
    return xy
end

mean_resultant_length(nrepetitions, nsteps, brw, crw, w) = norm(Folds.mapreduce(_ -> get_exit_point(nsteps, brw, crw, w), +, 1:nrepetitions; init=zero(SV)))/nrepetitions/nsteps

nsteps = 20
# brw_κ = Particles(Uniform(0.1, 400))
# brw = VonMises(brw_κ; check_args=false)
# crw_κ = Particles(Truncated(Normal(3.4, 0.3), 1e-6, Inf))
# crw = VonMises(crw_κ; check_args=false)

 

crw = Particles(rand(VonMises(3.4), 2000))
brw = Particles(rand(VonMises(400), 2000))
w = Particles(Uniform(0.3, 0.4))
xy = get_exit_point(nsteps, brw, crw, w)
r = norm(xy)/nsteps
using StatsPlots
plot(r)


function get_exit_point(nsteps, brw, crw, w)
    xy = zero(SV)
    θ = 0.0
    while norm(xy) < nsteps
        θ, Δ = next_step(θ, rand(brw), rand(crw), w)
        xy += Δ
    end
    return nsteps*normalize(xy) # extend the last step of the beetle so it crosses the nsteps circle around the origin
end

mean_resultant_length(nrepetitions, nsteps, brw::VonMises, crw::VonMises, w) = norm(Folds.mapreduce(_ -> get_exit_point(nsteps, brw, crw, w), +, 1:nrepetitions; init=zero(SV)))/nrepetitions/nsteps
mean_resultant_length(nrepetitions, nsteps, brw_κ::Real, crw_κ::Real, w) = mean_resultant_length(nrepetitions, nsteps, VonMises(brw_κ), VonMises(crw_κ), w)
mean_resultant_length(nrepetitions, nsteps, brw_κ::Real, crw::VonMises, w) = mean_resultant_length(nrepetitions, nsteps, VonMises(brw_κ), crw, w)



include("BeetleModel.jl")
using .BeetleModel

using MonteCarloMeasurements, Distributions
using SpecialFunctions
register_primitive(SpecialFunctions.besselix)

n = 200
brw_κ = Particles(n, Uniform(0.1, 400))
w = Particles(n, Uniform(0.01, 0.99))
nsteps = 20
crw_κ = 3.4
nrepetitions = 10
r = mean_resultant_length(nrepetitions, nsteps, brw_κ, crw_κ, w)

using MonteCarloMeasurements, SpecialFunctions, Distributions
x = Particles(Normal(1, 2))
y = Particles(Normal(2, 3))
besselix(x, y)
