using CSV, DataFrames
using Statistics, LinearAlgebra
using Distributions, StaticArrays
using GLMakie, Folds

const SV = SVector{2, Float64}

mean_resultant_length(degrees) = norm(Folds.mapreduce(SV ∘ reverse ∘ sincosd, +, degrees)/length(degrees)) 

df = CSV.read("semipunctatus.csv", DataFrame)
DataFrames.transform!(df, r"\d+" => ByRow(vcat) => :degrees, [:arena_diameter, :step_size] => ByRow((d, s) -> d/2s) => :nsteps)
DataFrames.transform!(df, :degrees => ByRow(mean_resultant_length) => :rvalue)
select!(df, [:id, :nsteps, :rvalue])

using Statistics, LinearAlgebra
using Distributions, StaticArrays
using GLMakie, Folds

using Random
Random.seed!(0)

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

function probability(nsteps, brw, crw, w, rvalue)
    n = 10000
    rvalues = Folds.map(i -> mean_resultant_length(10, nsteps, brw, crw, w), 1:n)
    d = fit(Beta, rvalues)
    p = pdf(d, rvalue)
    return p
end

_, nsteps, rvalue = first(eachrow(first(groupby(df, :id))))

n = 10
brw_κ = exp.(range(log(0.01), log(400), n))
brw = VonMises.(brw_κ)
crw_κ = 4 # equivalent to an "angular deviation" of 30°
crw = VonMises(crw_κ)
w = range(0, 1, n)
Random.seed!(0)
p = probability.(nsteps, brw, crw, w', rvalue)

surface(brw_κ, w, p, axis=(; type=Axis3, xlabel="kappa", ylabel="weight"))


function probability(nsteps, brw, crw, w)
    n = 10000
    rvalues = Folds.map(i -> mean_resultant_length(10, nsteps, brw, crw, w), 1:n)
    d = fit(Beta, rvalues)
    return d.α, d.β
end

n = 10
brw_κ = range(0.01, 5, n)
# brw_κ = exp.(range(log(0.01), log(400), n))
brw = VonMises.(brw_κ)
crw_κ = 4 # equivalent to an "angular deviation" of 30°
crw = VonMises(crw_κ)
w = range(0, 1, n)
ab = probability.(nsteps, brw, crw, w')

surface(brw_κ, w, first.(ab), axis=(; type=Axis3, xlabel="kappa", ylabel="weight"))

surface(brw_κ, w, last.(ab), axis=(; type=Axis3, xlabel="kappa", ylabel="weight"))


