using CSV, DataFrames
using Statistics, LinearAlgebra
using Distributions, StaticArrays
using GLMakie, Folds

const SV = SVector{2, Float64}

mean_resultant_length(degrees) = norm(Folds.mapreduce(SV ∘ reverse ∘ sincosd, +, degrees)/length(degrees)) 

df = CSV.read("semipunctatus.csv", DataFrame)
DataFrames.transform!(df, r"\d+" => ByRow(vcat) => :degrees, [:arena_diameter, :step_size] => ByRow((d, s) -> d/2s) => :nsteps)

for grp in groupby(df, :id)
    sort!(grp, :nsteps)
end

fig = Figure()
ax = Axis(fig[3,1], xlabel = "Steps (#)", ylabel = "Mean resultant length")
sl = IntervalSlider(fig[2, 1], range = 1:10, startvalues = (1, 10))
function fun(i1, i2, nsteps, degrees)
    x1, x2 = nsteps
    y1, y2 = [mean_resultant_length(d[i1:i2]) for d in degrees]
    return (Point2f(x1, y1), Point2f(x2, y2))
end
ls = map(sl.interval) do (i1, i2)
    combine(groupby(df, :id), [:nsteps,:degrees] => (nsteps,degrees) -> fun(i1, i2, nsteps, degrees)).nsteps_degrees_function
end
color = map(ls) do ls
    [splat(>)(last.(ps)) ? :red : :black for ps in ls]
end
linesegments!(ax, ls; color)
labeltext = lift(((i1, i2),) -> "$i1-$i2", sl.interval)
Label(fig[1, 1], labeltext, tellwidth = false)




DataFrames.transform!(df, :degrees => ByRow(mean_resultant_length) => :rvalue)

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

mean_resultant_length(nrepetitions, nsteps, brw, crw, w) = norm(Folds.mapreduce(_ -> get_exit_point(nsteps, brw, crw, w), +, 1:nrepetitions)/nrepetitions/nsteps) 

# TODO:
# build a cube
# smooth it  abit
# interpolate
# use that to find the best brw and w that fits with a pair of nsteps and rvalues
# repeat for each beetle
# plot the distributions of the resulting compass errors and weights
#

nrepetitions = 100_000
n = 20
nstepses = range(5, 50, n)
crw_κ = 4 # equivalent to an "angular deviation" of 30°
w = range(0, 1, 11)
brw_κ = exp.(range(log(0.01), log(400), 5))
r = Folds.map(κwn -> mean_resultant_length(nrepetitions, κwn.nsteps, VonMises(κwn.κ), VonMises(crw_κ), κwn.w), ((; w, κ, nsteps) for w in w, κ in brw_κ, nsteps in nstepses))




using Optim

nrepetitions = 10_000
crw_κ = 4
lower = [0.01, 0.0]
upper = [400, 1.0]
initial_x = [1.0, 0.5]
data = select(first(groupby(df, :id)), :nsteps, :rvalue)
function fun(brww)
    brw, w = brww
    s = 0.0
    for (nsteps, rvalue) in eachrow(data)
        r = mean_resultant_length(nrepetitions, nsteps, VonMises(brw), VonMises(crw_κ), w)
        s += abs2(r - rvalue)
    end
    return s
end
results = optimize(fun, lower, upper, initial_x, Fminbox(GradientDescent()), Optim.Options(time_limit = 10))


for grp in groupby(df, :id)
    sort!(grp, :nsteps)
    color = grp.r[1] < grp.r[2] ? :black : :red
    lines!(ax, grp.nsteps, grp.r; color)
end

save("semipunctatus.png", fig)


cmp = combine(groupby(df, :id)) do grp
    sort!(grp, :nsteps)
    (; Δ = only(diff(grp.r)))
end

mean(cmp.Δ)


cmp = combine(groupby(df, :id)) do grp
    sort!(grp, :nsteps)
    x1, x2 = grp.nsteps
    y1, y2 = grp.r
    a = (y2 - y1)/(x2 - x1)
    b = y1 - a*x1
    (; a, b)
end


fig = Figure()
ax = Axis(fig[1,1], xlabel = "Steps (#)", ylabel = "Mean resultant length")
for (k, grp) in pairs(groupby(df[15:20,:], :id))
    sort!(grp, :nsteps)
    lines!(ax, grp.nsteps, grp.r, label=k.id)
end
axislegend(ax)
