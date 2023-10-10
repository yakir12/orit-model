using Statistics, LinearAlgebra
using Distributions, StaticArrays
using DataFrames, AlgebraOfGraphics
using GLMakie, Folds

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

mean_resultant_length(nrepetitions, nsteps, brw, crw, w) = norm(Folds.mapreduce(_ -> get_exit_point(nsteps, brw, crw, w), +, 1:nrepetitions)/nrepetitions/nsteps) 

fwhm(κ) = rad2deg(2acos(log(cosh(κ))/κ)) # convert Von Mises concentration κ to full widith at half maximum in degrees


nrepetitions = 1000
crw_κ = 4 # equivalent to an "angular deviation" of 30°
n = 5

df = allcombinations(DataFrame, w = range(0, 1, n), brw_κ = exp.(range(log(0.01), log(400), n)), nsteps = [14, 27])
transform!(df, [:w, :brw_κ, :nsteps] => ByRow((w, brw_κ, nsteps) -> [mean_resultant_length(nrepetitions, nsteps, VonMises(brw_κ), VonMises(crw_κ), w) for _ in 1:1000]) => :r)
df = flatten(df, :r)

plt = data(df) * mapping(:r, color = :nsteps => string, dodge=:nsteps => string) * histogram(bins=20) * mapping(row=:w => string, col=:brw_κ => string)
fig, g = draw(plt, facet = (; linkxaxes = :none, linkyaxes = :none))
Label(fig[0,1], "Compass error (κ)")
Label(fig[1, 0], "Weight", rotation=pi/2)



