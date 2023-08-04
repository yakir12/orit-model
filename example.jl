# heatmap from article

n = 25
nsteps = 20
nrepetitions = 100
w = range(0, 1, n)
brw_σ = range(0, deg2rad(20), n + 2)[2:end-1] # compass error
crw_σ = deg2rad(30) # motor error
r = mean_resultant_length.(nrepetitions, nsteps, brw_σ', crw_σ, w)
fig = Figure()
ax = Axis(fig[1,1], xlabel="weight", ylabel="Compass (°)")
heatmap!(ax, w, rad2deg.(brw_σ), r)
Colorbar(fig[1,2], label="mean resultant length", limits=(0,1))

