# calculate a relationship between κ and the r-value. Let a beetle select an azimuth from a known Von Mises distribution (with concentration κ), take a step in that direction, and repeat until it passes a nsteps distance. Repeat for ntracks and calculate the r-value for the resulting exit azimuths. 
# CRW = Motor
# BRW = Compass

using Statistics, LinearAlgebra
using Distributions, StaticArrays, SpecialFunctions
using GLMakie, AlgebraOfGraphics
include("functions.jl")
include("runtests.jl")


# κ = 1
# d = VonMises(κ)
# α = -π:0.001:π
# p = pdf.(d, α)
# lines(α, p)


###############
# compare std2kappa with Orit
using DelimitedFiles

κ = [0.0000001,0.00001,0.0001,0.001, collect(0.1:0.1:699.9)...]
σ = std.(VonMises.(κ))
# lines(σ, κ, label = "Julia")
fig = Figure()
ax = Axis(fig[1,1], xlabel = "kappa", ylabel = "difference between matlab and julia std")
for n in (1, 10, 100, 500)
    matlab = vec(readdlm("$n.csv"))
    lines!(ax, κ, σ .- matlab, label = string(1000n))
end
axislegend("Number of samples from randpdf")
save("compare.png", fig)



###############
# cool widget that shows how Orit's model works

ϵ = 0.01
fig = Figure()
sg = SliderGrid(fig[1, 1:4],
                (label = "motor std", range = range(ϵ, 1 - ϵ, 100), startvalue=deg2rad(30), format=x -> string(round(rad2deg(x), digits = 1))),
                (label = "compass std", range = range(ϵ, 1 - ϵ, 100), startvalue=deg2rad(6), format=x -> string(round(rad2deg(x), digits = 1))),
                (label = "weight", range = range(0, 1, 100), startvalue=0.8),
                (label = "n steps", range = 10:10:50, startvalue=20),
                tellheight=false
               )
crw_σ = sg.sliders[1].value
crw = @lift VonMises(0, std2κ($crw_σ))
ax1 = Axis(fig[2,1], xlabel="θ (°)", ylabel="Probability", aspect=AxisAspect(1), limits=((-180, 180), (0, 5)))
x = range(-π, π, 101)
lines!(ax1, rad2deg.(x), @lift(pdf.($crw, x)), label="motor")
brw_σ = sg.sliders[2].value
brw = @lift VonMises(0, std2κ($brw_σ))
lines!(ax1, rad2deg.(x), @lift(pdf.($brw, x)), label="compass")
axislegend(ax1)
nsteps = sg.sliders[4].value
w = sg.sliders[3].value
ntracks = 1000
tracks = [@lift(get_track($nsteps, $brw, $crw, $w)) for _ in 1:ntracks]
ax2 = Axis(fig[2,2], aspect=DataAspect(), xlabel="X (step)", ylabel="Y (step)")
on(nsteps) do nsteps
    ax2.limits[] = ((-nsteps, nsteps), (-nsteps, nsteps))
end
for i in 1:ntracks
    lines!(ax2, tracks[i])
end
lines!(ax2, @lift(Circle(zero(Point2f), $nsteps)), color=:black)
ax3 = Axis(fig[2,3], xlabel="exit azimuth (°)", ylabel="#", limits=((-180, 180), (0, ntracks)), aspect=AxisAspect(1))
azimuths = lift(tracks[1]) do _
    [atan(last(track[])...) - π/2 for track in tracks]
end
hist!(ax3, @lift rad2deg.($azimuths))
r = @lift mean_resultant_length($azimuths)
Label(fig[2,4], @lift(string("r-value = ", round($r, digits=4))), tellheight=false)

##################

# r-value distribution for lamarcki

ϵ = 1e-6
n_individuals = 20
n_repetitions = 20
brw_σ = truncated(Normal(deg2rad(6.34), deg2rad(3.01)), ϵ, sqrt(2) - ϵ)
crw_σ = truncated(Normal(deg2rad(29.41), deg2rad(9.92)), ϵ, sqrt(2) - ϵ)
w_σ = truncated(Normal(0.83, 0.08), 0, 1)
nsteps = 20
rs = [mean_resultant_length(n_repetitions, nsteps, brw_σ, crw_σ, w_σ) for _ in 1:n_individuals]
boxplot(ones(n_individuals), rs, axis=(;limits=((nothing, nothing), (0.6, 1.0))))

###############
#analytical relationships

n = 1000
σ = range(0, sqrt(2), n + 2)[2:end-1]
κ = std2κ.(σ)
d = VonMises.(κ)
r = mean_resultant_length.(d)
lines(rad2deg.(σ), r, axis=(; xlabel="std (°)", ylabel="mean resultant length"))

###########
# w = 0 removes any infuence from the compass

using ProgressMeter, DataFrames

n_repetitions = 10000
nsteps = 20
n = 10
ϵ = 0.001
nw = 10
df = DataFrame((; brw_σ, crw_σ, weight) for brw_σ in range(ϵ, sqrt(2) - ϵ, n) for crw_σ in range(ϵ, sqrt(2) - ϵ, n) for weight in range(0, 1, nw))
N = nrow(df)
df.r .= zeros(N)
p = Progress(N)
Threads.@threads for i in 1:N
    df.r[i] = mean_resultant_length.(n_repetitions, nsteps, df[i, :brw_σ], df[i, :crw_σ], df[i, :weight])
    next!(p)
end
plt = data(df) * mapping(:weight, :r => "r-value", row=:brw_σ => nonnumeric ∘ (x -> round(Int, rad2deg(x))), col=:crw_σ => nonnumeric ∘ (x -> round(Int, rad2deg(x)))) * visual(Lines)
fig = Figure()
Label(fig[1,1], "Motor (°)", tellheight=true, tellwidth=false)
draw!(fig[2,1], plt)#, axis=(bottomspinevisible=false, topspinevisible=false, leftspinevisible=false, rightspinevisible=false, xgridvisible=false, ygridvisible=false))
Label(fig[2,2], "Compass (°)", tellheight=false, tellwidth=true, rotation=-π/2)

############
# try to add nsteps as a factor 

using ProgressMeter, DataFrames

ϵ = 0.01
crw_σ = truncated(Normal(deg2rad(30), deg2rad(10)), ϵ, 1 - ϵ)
crw_σ_min, crw_σ_max = quantile(crw_σ, [0.05, 0.95])
n = 10
n_repetitions = 1000
nw = 10
df = DataFrame((; brw_σ, crw_σ, weight, nsteps) for brw_σ in range(ϵ, 1 - ϵ, n) for crw_σ in range(crw_σ_min, crw_σ_max, n) for weight in range(0, 1, nw), nsteps in (20, 40))
N = nrow(df)
df.r .= zeros(N)
p = Progress(N)
Threads.@threads for i in 1:N
    df.r[i] = mean_resultant_length.(n_repetitions, df[i, :nsteps], df[i, :brw_σ], df[i, :crw_σ], df[i, :weight])
    next!(p)
end

plt = data(df) * mapping(:weight, :r => "r-value", row=:brw_σ => nonnumeric ∘ (x -> round(Int, rad2deg(x))), col=:crw_σ => nonnumeric ∘ (x -> round(Int, rad2deg(x))), color = :nsteps) * visual(Scatter) * mapping(color = :nsteps)
fig = Figure()
Label(fig[1,1], "Motor (°)", tellheight=true, tellwidth=false)
draw!(fig[2,1], plt)#, axis=(bottomspinevisible=false, topspinevisible=false, leftspinevisible=false, rightspinevisible=false, xgridvisible=false, ygridvisible=false))
Label(fig[2,2], "Compass (°)", tellheight=false, tellwidth=true, rotation=-π/2)


##################
# heatmap

n_repetitions = 100
n = 10
brw_σ = range(0, sqrt(2), n + 2)[2:end-1]
crw_σ = range(0, sqrt(2), n + 2)[2:end-1]
fig = Figure()
sg = SliderGrid(fig[1, 1:2],
                (label = "weight", range = range(0, 1, 100), startvalue=0.35),
                (label = "n steps", range = 10:50, startvalue=20),
                tellheight=false
               )
w = sg.sliders[1].value
nsteps = sg.sliders[2].value
ax = Axis(fig[2,1], xlabel="Compass (°)", ylabel="Motor (°)")#, aspect=AxisAspect(1))
r = @lift mean_resultant_length.(n_repetitions, $nsteps, brw_σ, crw_σ', $w)
heatmap!(ax, rad2deg.(brw_σ), rad2deg.(crw_σ), r)
Colorbar(fig[2,2], label="mean resultant length", limits=(0,1))


n_repetitions = 10000
n = 10
brw_σ = range(0, sqrt(2), n + 2)[2:end-1]
crw_σ = range(0, sqrt(2), n + 2)[2:end-1]
fig = Figure()
w = 0.192
nsteps = 20
ax = Axis(fig[1,1], xlabel="Motor (°)", ylabel="Compass (°)")#, aspect=AxisAspect(1))
r = mean_resultant_length.(n_repetitions, nsteps, brw_σ, crw_σ', w)
heatmap!(ax, rad2deg.(brw_σ), rad2deg.(crw_σ), r)
Colorbar(fig[1,2], label="mean resultant length", limits=(0,1))

#########################
# heatmap from article
using ProgressMeter

n = 100
nsteps = 20
nrepetitions = 100000
w = range(0, 1, n)
brw_σ = range(0, deg2rad(rad2deg(sqrt(2))), n + 2)[2:end-1] # compass error
crw_σ = deg2rad(30) # motor error
r = Array{Float64}(undef, n, n)
p = Progress(n^2)
Threads.@threads for k in 1:n^2
    i, j = Tuple(CartesianIndices((n, n))[k])
    r[i, j] = mean_resultant_length(nrepetitions, nsteps, brw_σ[j], crw_σ, w[i])
    next!(p)
end
# r = mean_resultant_length.(nrepetitions, nsteps, brw_σ', crw_σ, w)
fig = Figure()
ax = Axis(fig[1,1], xlabel="weight", ylabel="Compass (°)")
heatmap!(ax, w, rad2deg.(brw_σ), r)
Colorbar(fig[1,2], label="mean resultant length", limits=(0,1))
i = [findfirst(<(0.9), row) for row in eachrow(r)]
lines!(ax, w, rad2deg.(brw_σ[i]))
save("figure 4.png", fig)


fig = Figure()
ax = Axis(fig[1,1], xlabel="weight", ylabel="Compass (°)")
contour!(ax, w, rad2deg.(brw_σ), r, levels=0:0.1:1, labels=true)
save("figure 4 contour lines.png", fig)
