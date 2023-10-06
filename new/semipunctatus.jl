using CSV, DataFrames
using Statistics, LinearAlgebra
using Distributions, StaticArrays
using GLMakie, Folds

const SV = SVector{2, Float64}

mean_resultant_length(degrees) = norm(Folds.mapreduce(SV ∘ reverse ∘ sincosd, +, degrees)/length(degrees)) 

df = CSV.read("semipunctatus.csv", DataFrame)
transform!(df, r"\d+" => ByRow(vcat) => :degrees, [:arena_diameter, :step_size] => ByRow((d, s) -> d/2s) => :nsteps)

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
