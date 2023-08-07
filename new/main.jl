# calculate a relationship between κ and the r-value. Let a beetle select an azimuth from a known Von Mises distribution (with concentration κ), take a step in that direction, and repeat until it passes a nsteps distance. Repeat for ntracks and calculate the r-value for the resulting exit azimuths. 
# CRW = Motor
# BRW = Compass

using Statistics, LinearAlgebra
using Distributions, StaticArrays, SpecialFunctions, ProgressMeter
using GLMakie, Folds, Transducers

include("functions.jl")

nsteps = 20
crw_κ = 4 # equivalent to an "angular deviation" of 30°
nrepetitions = 10000
n = 100

w = range(0, 1, n)
brw_κ = range(0.01, 10, n)
r = Folds.map(κw -> mean_resultant_length(nrepetitions, nsteps, κw.κ, crw_κ, κw.w), ((; κ, w) for w in w, κ in brw_κ))

fig = Figure()
ax = Axis(fig[1,1], xlabel="Weight", ylabel="Compass error κ")
heatmap!(ax, w, brw_κ, r)
Colorbar(fig[1,2], label="Mean resultant length", limits=(0,1))

save("figure 4.png", fig)

