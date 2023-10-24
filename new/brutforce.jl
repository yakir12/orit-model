include("BeetleModel.jl")

using .BeetleModel
using GLMakie, Folds, Distributions, Interpolations, KernelDensity

function get_likelihood(nrepetitions, nsteps, brw, crw, w, n, rvalue)
    rvalues = Folds.map(i -> mean_resultant_length(nrepetitions, nsteps, brw, crw, w), 1:n)
    k = kde(rvalues)
    return pdf(k, rvalue)
end



function fitBeta(nrepetitions, nsteps, brw, crw, w, n)
    rvalues = Folds.map(i -> mean_resultant_length(nrepetitions, nsteps, brw, crw, w), 1:n)
    d = fit(Beta, rvalues)
    return (; α = d.α, β = d.β, ll = loglikelihood(d, rvalues))
end

ndata = 15

nrepetitions = 10
nsteps = range(10, 35, ndata)
brw_κ = range(0.1, 100, ndata)
crw_κ = 3.4
w = range(0, 1, ndata)
n = 10_000

using Serialization
data = deserialize("data")

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
select!(df, [:id, :nsteps, :rvalues])
# subset!(df, :id => ByRow(!=("A.semip_09")))
# nsteps = df.nsteps
# rvalues = df.rvalues

function likelihood(rvalues, nsteps, α, β, brw_κ::Float64, w::Float64)
    p = 0.0
    for (rvalue, nstep) in zip(rvalues, nsteps)
        d = Beta(α[nstep, brw_κ, w], β[nstep, brw_κ, w])
        p += logpdf(d, rvalue)
    end
    exp(p)
end


using Optim

function optimize2rvalues(rvalues, nsteps)
    lower = first.([brw_κ, w]) .+ 1e-3
    upper = last.([brw_κ, w]) .- 1e-3
    initial_x = (lower .+ upper)/2
    inner_optimizer = GradientDescent()
    results = optimize(x -> 1/likelihood(rvalues, nsteps, α, β, x...), lower, upper, initial_x, Fminbox(inner_optimizer))
    return (; brw_κ = results.minimizer[1], w = results.minimizer[2])
end
res = combine(groupby(df, :id), [:rvalues, :nsteps] => optimize2rvalues => [:brw_κ, :w]; keepkeys=false)

brw_κ, w = mean.(eachcol(res))

scatter(res.brw_κ, res.w, axis=(; xlabel="Compass error (κ)", ylabel="Weight", limits=((0, 70), (0, 1))))

n = 1000
brw_κ = range(0.1, 100, n)
w = range(0, 1, n)
l = ones(n , n)
for grp in groupby(df, :id)
    x = [likelihood(grp.rvalues, grp.nsteps, α, β, brw_κ, w) for brw_κ in brw_κ, w in w]
    l .*= x#/sum(x)
end
heatmap(brw_κ, w, l)

fig = Figure()
for (i, grp) in enumerate(groupby(df, :id))
    l = [likelihood(grp.rvalues, grp.nsteps, α, β, brw_κ, w) for brw_κ in brw_κ, w in w]
    ax = Axis(fig[i, 1])
    heatmap!(ax, brw_κ, w, l)
end

