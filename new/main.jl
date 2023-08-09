# calculate a relationship between κ and the r-value. Let a beetle select an azimuth from a known Von Mises distribution (with concentration κ), take a step in that direction, and repeat until it passes a nsteps distance. Repeat for ntracks and calculate the r-value for the resulting exit azimuths. 
# CRW = Motor
# BRW = Compass

using Statistics, LinearAlgebra
using Distributions, StaticArrays
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
    return xy
end

mean_resultant_length(nrepetitions, nsteps, brw, crw, w) = norm(Folds.mapreduce(_ -> get_exit_point(nsteps, brw, crw, w), +, 1:nrepetitions)/nrepetitions/nsteps) 

fwhm(κ) = rad2deg(2acos(log(cosh(κ))/κ)) # convert Von Mises concentration κ to full widith at half maximum in degrees

# figure 4:

nrepetitions = 100_000
nsteps = 20
crw_κ = 4 # equivalent to an "angular deviation" of 30°
n = 100

w = range(0, 1, n)
brw_κ = range(0.01, 400, n)
r = Folds.map(κw -> mean_resultant_length(nrepetitions, nsteps, VonMises(κw.κ), VonMises(crw_κ), κw.w), ((; κ, w) for w in w, κ in brw_κ))

fig = Figure()
ax = Axis(fig[1,1], xlabel="Weight", ylabel="Compass error κ")
heatmap!(ax, w, brw_κ, r)
Colorbar(fig[1,2], label="Mean resultant length", limits=(0,1))

save("figure 4.png", fig)

# better figure 4:

using Contour

c = contours(w, fwhm.(brw_κ), r, [0.8, 0.9, 0.95, 0.99])
fig = Figure()
ax = Axis(fig[1,1], xlabel="Weight", ylabel="Compass FWHM", xticks = (0:0.5:1, ["only motor", "0.5", "only compass"]), ytickformat = "{:d}°", limits = ((nothing, 1.1), (nothing, 100)), title="Mean resultant vector length")
crw_fwhm = fwhm(crw_κ)
scatter!(ax, 0, 20, color=(:black, 0))
for cl in levels(c)
    lvl = level(cl)
    line = only(Contour.lines(cl))
    xs, ys = coordinates(line)
    lines!(ax, xs, ys, color=[lvl], colorrange=(0.75, 0.99))
    _, i = findmax(xs)
    x = xs[i]
    y = ys[i]
    text!(ax, x, y, text=string(lvl), align=(:left,:center), offset = (5, 0))
end
ax2 = Axis(fig[1,1], yticks = ([crw_fwhm], ["motor\nFWHM"]), yaxisposition = :right)
linkyaxes!(ax, ax2)
hidespines!(ax2)
hidexdecorations!(ax2)
ax3 = Axis(fig[1,1], yticks = ([crw_fwhm], [string(round(Int, crw_fwhm), "°")]))
linkyaxes!(ax, ax3)
hidespines!(ax3)
hidexdecorations!(ax3)


save("contour.png", fig)
