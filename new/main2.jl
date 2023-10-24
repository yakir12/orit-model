# calculate a relationship between κ and the r-value. Let a beetle select an azimuth from a known Von Mises distribution (with concentration κ), take a step in that direction, and repeat until it passes a nsteps distance. Repeat for ntracks and calculate the r-value for the resulting exit azimuths. 
# CRW = Motor
# BRW = Compass

include("BeetleModel.jl")

using .BeetleModel
using GLMakie, Folds


# figure 4:

nrepetitions = 100_000
nsteps = 20
crw_κ = 3.4 # equivalent to an "angular deviation" of 30°
n = 25

w = range(0, 1, n)
brw_κ = exp.(range(log(0.01), log(400), n - 1))
push!(brw_κ, crw_κ)
sort!(brw_κ)
r = Folds.map(κw -> mean_resultant_length(nrepetitions, nsteps, κw.κ, crw_κ, κw.w), ((; κ, w) for w in w, κ in brw_κ))

fig = Figure()
ax = Axis(fig[1,1], xlabel="Weight", ylabel="Compass error κ")
heatmap!(ax, w, brw_κ, r)
Colorbar(fig[1,2], label="Mean resultant length", limits=(0,1))

save("figure 4.png", fig)


using Contour

fwhm(κ) = rad2deg(2acos(log(cosh(κ))/κ)) # convert Von Mises concentration κ to full widith at half maximum in degrees

c = contours(w, fwhm.(brw_κ), r, [0.8, 0.9, 0.95, 0.99])
fig = Figure()
ax = Axis(fig[1,1], xlabel="Weight", ylabel="Compass FWHM", xticks = (0:0.5:1, ["only motor", "0.5", "only compass"]), ytickformat = "{:d}°", limits = ((nothing, 1.1), (nothing, 200)), title="Mean resultant vector length")
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

