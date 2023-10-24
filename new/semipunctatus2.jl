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
subset!(df, :id => ByRow(!=("A.semip_09")))

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


crw_κ = 4
crw = VonMises(crw_κ)
grps = groupby(df, :id)
function probability(nsteps, brw, w, rvalue)
    n = 100000
    rvalues = Folds.map(i -> mean_resultant_length(10, nsteps, brw, crw, w), 1:n)
    d = fit(Beta, rvalues)
    p = logpdf(d, rvalue)
    return p
end
function probability(brw_κ, w)
    brw = VonMises(brw_κ)
    p = 0.0
    for grp in grps
        for (; nsteps, rvalue) in eachrow(grp)
            p += probability(nsteps, brw, w, rvalue)
        end
    end
    return exp(p)
end

@btime probability(6, 0.2)

n = 20
brw_κ = range(0.5, 40, n)
w = range(0.1, 0.3, n)
p = probability.(brw_κ, w')
heatmap(brw_κ, w, p, axis=(; xlabel="Compass error (κ)", ylabel="Weight"))



using Optim
lower = [0.5, 0.15]
upper = [40, 0.3]
initial_x = [6, 0.2]
inner_optimizer = GradientDescent()
results = optimize(Base.Fix2(\, 1) ∘ splat(probability), lower, upper, initial_x, Fminbox(inner_optimizer))




d = Normal(2,0.5)
xs = range(0, 4, 101)
lines(xs, cdf.(d, xs))

x = 1.3
p = pdf(d, x)
n = 10000000
xs = rand(d, n);
sort!(xs);
hist!(xs, bins=100, normalization=:pdf)
findfirst(>(x), xs)/n



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
    n = 1000000
    rvalues = Folds.map(i -> mean_resultant_length(10, nsteps, brw, crw, w), 1:n)
    d = fit(Beta, rvalues)
    return d.α, d.β
end

n = 25
# brw_κ = range(0.01, 5, n)
brw_κ = exp.(range(log(0.01), log(400), n))
brw = VonMises.(brw_κ)
crw_κ = 4 # equivalent to an "angular deviation" of 30°
crw = VonMises(crw_κ)
w = range(0, 1, n)
ab = probability.(nsteps, brw, crw, w')

@btime probability(nsteps, brw[1], crw, w[1])

surface(brw_κ, w, first.(ab), axis=(; type=Axis3, xlabel="kappa", ylabel="weight"))

surface(brw_κ, w, last.(ab), axis=(; type=Axis3, xlabel="kappa", ylabel="weight"))


