using Statistics, LinearAlgebra
using Distributions, StaticArrays
using DataFrames, AlgebraOfGraphics
using GLMakie, Folds
using CategoricalArrays

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


nrepetitions = 100_000
crw_κ = 4 # equivalent to an "angular deviation" of 30°
n = 5

w = range(0, 1, n)
brw_κ = round.(exp.(range(log(0.01), log(400), n)), digits=2)
df = allcombinations(DataFrame, w =w, brw_κ = brw_κ, nsteps = range(14, 27, n))
transform!(df, [:w, :brw_κ, :nsteps] => ByRow((w, brw_κ, nsteps) -> mean_resultant_length(nrepetitions, nsteps, VonMises(brw_κ), VonMises(crw_κ), w)) => :r)


plt = data(df) * mapping(:nsteps => "# steps", :r => "Mean resultant length") * visual(Lines) * mapping(row=:w => sorter(reverse(w)), col=:brw_κ => nonnumeric)
fig, g = draw(plt)
Label(fig[0, :], "Compass error (κ)", fontsize=20, font=:bold)
Label(fig[1:end, end + 1], "Weight", rotation = -π/2, fontsize=20, font=:bold)

save("nsteps.png", fig)

ratios = combine(groupby(subset(df, :nsteps => ByRow(∈((14, 27)))), [:w, :brw_κ])) do grp
    sort!(grp, :nsteps)
    (; ratio = /(grp.r...))
end
img = Matrix{Float64}(undef, n, n)
for (wi, brw_κi, ratio) in eachrow(ratios)
    i = findfirst(==(wi), w)
    j = findfirst(==(brw_κi), brw_κ)
    img[j, i] = ratio
end
fig = Figure()
ax = Axis(fig[1,1], xlabel = "Compass error (κ)", xticks = (1:n, string.(brw_κ)), ylabel = "Weight", yticks = (1:n, string.(w)))
b = maximum(x -> abs(1 - x), extrema(img))
colorrange = (1-b, 1+b)
hm = heatmap!(ax, img; colorrange, colormap = :bluesreds)
for ind in eachindex(IndexCartesian(), img)
    i, j = Tuple(ind)
    text!(ax, i, j; text = string(round(img[ind], digits=2)), align=(:center, :center))
end
Colorbar(fig[1,2], hm, label = "Ratio (14 steps / 27 steps)")

save("ratios.png", fig)







nrepetitions = 10
df = allcombinations(DataFrame, w = w, brw_κ = brw_κ, nsteps = [14, 27])
transform!(df, [:w, :brw_κ, :nsteps] => ByRow((w, brw_κ, nsteps) -> [mean_resultant_length(nrepetitions, nsteps, VonMises(brw_κ), VonMises(crw_κ), w) for _ in 1:1000]) => :r)
df = flatten(df, :r)

plt = data(df) * mapping(:r => "Mean resultant length", color = :nsteps => nonnumeric => "# steps") * AlgebraOfGraphics.density() * mapping(row=:w => sorter(reverse(w)), col=:brw_κ => nonnumeric)
fig, g = draw(plt, facet = (; linkxaxes = :none, linkyaxes = :none))
Label(fig[0, :], "Compass error (κ)", fontsize=20, font=:bold)
Label(fig[1:end, end + 1], "Weight", rotation = -π/2, fontsize=20, font=:bold)


save("pdf.png", fig)




n = 100000
nrepetitions = 10_000
l = 0.01
u = 400
ll = log(l)
lu = log(u)
w = rand(n)
df = DataFrame(w = Float64[], brw_κ = Float64[], nsteps = Int[], type = String[])
for (type, brw_κ) in zip(("linear", "logarithmic"), (l .+ (u - l) .* rand(n), exp.(ll .+ (lu - ll) .* rand(n)))), nsteps in (14, 27)
    df1 = DataFrame(w = w, brw_κ = brw_κ, nsteps = nsteps, type = type)
    append!(df, df1)
end
transform!(df, [:w, :brw_κ, :nsteps] => ByRow((w, brw_κ, nsteps) -> mean_resultant_length(nrepetitions, nsteps, VonMises(brw_κ), VonMises(crw_κ), w)) => :r)
ratios = combine(groupby(df, [:w, :brw_κ, :type]), keepkeys = false) do grp
    sort!(grp, :nsteps)
    (; r = first(grp.r), ratio = /(grp.r...), type = first(grp.type) == "linear" ? 1 : 2)
end

cutoff = 0.8
sratios = subset(ratios, :r => ByRow(≥(cutoff)))
fig = Figure()
ax = Axis(fig[1,1], xlabel = "Mean resultant length", ylabel="Normalization ratio")#, limits = ((0.81,1+1/2nbins), nothing))
hlines!(ax, 1, color = :gray)
nbins = 10
fmt(from, to, i; leftclosed, rightclosed) = string((to - from)/2 + from)
boxplot!(ax, [parse(Float64, unwrap(x)) for x in cut(sratios.r, range(cutoff, 1, nbins + 1), labels = fmt)], sratios.ratio, width = 0.2/nbins, dodge = sratios.type, color=sratios.type, show_outliers = false, show_median = true)
a, b = [sratios.r ones(nrow(sratios))] \ sratios.ratio
ablines!(ax, b, a, color = :red, linewidth = 4)

save("distribution.png", fig)
