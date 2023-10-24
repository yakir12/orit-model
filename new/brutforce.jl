include("BeetleModel.jl")

using .BeetleModel
using GLMakie, Folds, Distributions, Interpolations


function fitBeta(nrepetitions, nsteps, brw, crw, w, n)
    rvalues = Folds.map(i -> mean_resultant_length(nrepetitions, nsteps, brw, crw, w), 1:n)
    d = fit(Beta, rvalues)
    return (α = d.α, β = d.β)
end

function likelihood(rvalues, nsteps, α, β, brw_κ::Float64, w::Float64)
    p = 0.0
    for (rvalue, nstep) in zip(rvalues, nsteps)
        d = Beta(α[nstep, brw_κ, w], β[nstep, brw_κ, w])
        p += logpdf(d, rvalue)
    end
    exp(p)
end

ndata = 15

nrepetitions = 10
nsteps = range(10, 35, ndata)
brw_κ = range(10, 70, ndata)
crw_κ = 3.4
w = range(0.05, 0.3, ndata)
n = 100_000

data = [fitBeta(nrepetitions, nsteps, brw_κ, crw_κ, w, n) for nsteps in nsteps, brw_κ in brw_κ, w in w]

α = cubic_spline_interpolation((nsteps, brw_κ, w), getfield.(data, :α))
β = cubic_spline_interpolation((nsteps, brw_κ, w), getfield.(data, :β))
# α = linear_interpolation((nsteps, brw_κ, w), getfield.(data, :α))
# β = linear_interpolation((nsteps, brw_κ, w), getfield.(data, :β))

using DataFrames, CSV, StaticArrays, LinearAlgebra
const SV = SVector{2, Float64}

mean_resultant_length2(degrees) = norm(Folds.mapreduce(SV ∘ reverse ∘ sincosd, +, degrees))/length(degrees) 
df = CSV.read("semipunctatus.csv", DataFrame)
DataFrames.transform!(df, r"\d+" => ByRow(vcat) => :degrees, [:arena_diameter, :step_size] => ByRow((d, s) -> d/2s) => :nsteps)
DataFrames.transform!(df, :degrees => ByRow(mean_resultant_length2) => :rvalues)
# subset!(df, :id => ByRow(!=("A.semip_09")))
nsteps = df.nsteps
rvalues = df.rvalues

n = 100
brw_κ = range(10, 70, n)
w = range(0.05, 0.3, n)
l = [likelihood(rvalues, nsteps, α, β, brw_κ, w) for brw_κ in brw_κ, w in w]
l ./= sum(l)
heatmap(brw_κ, w, l)
