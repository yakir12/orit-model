include("BeetleModel.jl")

using .BeetleModel
using GLMakie, Folds, Distributions, Interpolations, KernelDensity, Optim



# function sample!(α, nsteps, brw, crw, w, nrepetitions)
#     map!(_-> get_exit_azimuth(nsteps, brw, crw, w), α, 1:nrepetitions)
#     sort!(α)
#     α .-= α[1]
#     return α
# end
#
#
# nsteps, brw, crw, w, nrepetitions = (20, VonMises(20), VonMises(3.4), 0.5, 10)
# s1 = Vector{Int}(undef, nrepetitions)
# sample!(s1, nsteps, brw, crw, w, nrepetitions)
#
# s2 = Vector{Int}(undef, nrepetitions)
# n = 1_000_000
#
# res = optimize(w -> -Folds.count(sample!(s2, nsteps, brw, crw, w, nrepetitions) == s1 for _ in 1:n), 0, 1)
#
# nsteps, brw, crw, w, nrepetitions = (20, VonMises(20), VonMises(3.4), 0.5, 10)
# ss1 = map(1:10) do _
#     s = Vector{Int}(undef, nrepetitions)
#     sample!(s, nsteps, brw, crw, w, nrepetitions)
#     return copy(s)
# end
#
# brw_κ = exp.(range(log(0.1), log(30), 25))
# w = similar(brw_κ)
# p = similar(w)
# for (i, brw) in enumerate(brw_κ)
#     res = optimize(wi -> -Folds.count(sample!(s2, nsteps, VonMises(brw), crw, wi, nrepetitions) ∈ ss1 for _ in 1:n), 0, 1)
#     w[i] = res.minimizer
#     p[i] = -res.minimum
# end
# lines(w, brw_κ, color = p)
#
# w = range(0, 1, 25)
# p = [Folds.count(sample!(s2, nsteps, brw, crw, w, nrepetitions) == s1 for _ in 1:n)/n for w in w]
# lines(w, p)
#
# @time Folds.count(sample!(s2, nsteps, brw, crw, w, nrepetitions) == s1 for _ in 1:n)/n
#
# n = 10 .^ (1:7)
# p = [Folds.count(sample!(s2, nsteps, brw, crw, w, nrepetitions) == s1 for _ in 1:i)/i for i in n]
#
# lines(n, p)
#
# c = 0
# # d = 0.0
# for _ in 1:n
#     sample!(s2, nsteps, brw, crw, w, nrepetitions)
#     if s2 == s1
#         c += 1
#     end
# end
# c/n

n = 20
nrepetitions = 100_000
nsteps = 20
# brw_κs = range(1, 20, n)
brw_κs = exp10.(range(log10(0.1), log10(100), n))
brws = VonMises.(brw_κs)
crw_κ = 4
crw = VonMises(crw_κ)
ws = range(0, 1, n)
p = Folds.collect(mean_resultant_length(nrepetitions, nsteps, brw, crw, w) for w in ws, brw in brws)

wbrw = [(ws[first(Tuple(ind))], brw_κs[last(Tuple(ind))]) for ind in CartesianIndices((n, n)) if 0.9 < p[ind] < 0.95]

fig = Figure()
ax = Axis(fig[1,1], limits = ((0,1), extrema(brw_κs)), xlabel="Weight", ylabel="Compass error (κ)", yscale=log10)
contour!(ax, ws, brw_κs, p; levels=[0.9, 0.95, 0.99], labels=true, color=:black)
scatter!(ax, wbrw)





function get_likelihood(nrepetitions::Int, nsteps, brw::Real, crw::Real, w::Real, n::Int, rvalues)
    p = 0.0
    for (rvalue, nstep) in zip(rvalues, nsteps)
        x = Folds.map(i -> mean_resultant_length(nrepetitions, nstep, brw, crw, w), 1:n)
        k = kde(x)
        p1 = pdf(k, rvalue)
        p += p1 < 0 ? 0.0 : log(p1)
    end
    return p/length(rvalues)
end

using DataFrames, CSV, StaticArrays, LinearAlgebra
const SV = SVector{2, Float64}

nrepetitions, nsteps, brw_κ, crw_κ, w = (10, 20, 20, 4, 0.3)
rvalues = [mean_resultant_length(nrepetitions, nsteps, VonMises(brw_κ), VonMises(crw_κ), w) for _ in 1:100]
nstepss = fill(nsteps, length(rvalues))
ndata = 20
brw_κs = exp.(range(log(1), log(30), ndata))
ws = range(0, 1, ndata)
n = 1_000
p = [get_likelihood(nrepetitions, nstepss, brw_κ, crw_κ, w, n, rvalues) for w in ws, brw_κ in brw_κs]
heatmap(ws, brw_κs, exp.(p))
scatter!(w, brw_κ) 

ws = Vector{Float64}(undef, ndata)
p = Vector{Float64}(undef, ndata)
for (i, brw_κ) in enumerate(brw_κs)
    r = optimize(w -> -get_likelihood(nrepetitions, nstepss, brw_κ, crw_κ, w, n, rvalues), 0, 1, Brent(); abs_tol=0.005)
    ws[i] = r.minimizer
    p[i] = -r.minimum
end
lines(ws, brw_κs, color = p, axis=(;limits=((0,1), extrema(brw_κs))))
scatter!(w, brw_κ) 





mean_resultant_length2(degrees) = norm(Folds.mapreduce(SV ∘ reverse ∘ sincosd, +, degrees))/length(degrees) 
df = CSV.read("semipunctatus.csv", DataFrame)
DataFrames.transform!(df, r"\d+" => ByRow(vcat) => :degrees, [:arena_diameter, :step_size] => ByRow((d, s) -> d/2s) => :nsteps)
DataFrames.transform!(df, :degrees => ByRow(mean_resultant_length2) => :rvalues)
select!(df, [:id, :nsteps, :rvalues])

# df = df[1:6, :]

# ndata = 20
# nrepetitions = 10
# brw_κ = range(0.1, 30, ndata)
# # brw_κ = exp.(range(log(0.1), log(30), ndata))
# crw_κ = 3.4
# w = range(0, 1, ndata)
# n = 10_000
#
# l = [[get_likelihood(nrepetitions, grp.nsteps, brw_κ, crw_κ, w, n, grp.rvalues) for brw_κ in brw_κ, w in w] for grp in groupby(df, :id)]
#
#
# fig = Figure()
# for (i, (l, grp)) in enumerate(zip(l, groupby(df, :id)))
#     ax = Axis(fig[i, 1], title=string(grp.id[1]))
#     heatmap!(ax, brw_κ, w, l)
# end
#
# heatmap(brw_κ, w, reduce(*, l))
#
#
#
# grp = first(groupby(df, :id))
#
# using ProgressMeter
ndata = 20
nrepetitions = 10
# brw_κ = range(0.1, 30, ndata)
brw_κ = exp.(range(log(0.1), log(30), ndata))
crw_κ = 3.4
n = 100_000
res = combine(groupby(df, :id)) do grp
    w = Vector{Float64}(undef, ndata)
    p = Vector{Float64}(undef, ndata)
    for (i, brw_κ) in enumerate(brw_κ)
        r = optimize(w -> -get_likelihood(nrepetitions, grp.nsteps, brw_κ, crw_κ, w, n, grp.rvalues), 0, 1, Brent(); abs_tol=0.005)
        w[i] = r.minimizer
        p[i] = -r.minimum
    end
    (; w, p)
end


fig = Figure()
ax = Axis(fig[1,1], xlabel="Weight", ylabel="Compass error (κ)", limits=((0, 1), extrema(brw_κ)))
for grp in groupby(res, :id)
    y = copy(grp.w)
    for i in eachindex(y)
        y[i] = grp.p[i] < 10 ? NaN : y[i]
    end
    lines!(ax, y, brw_κ, color=:black)
end
hlines!(ax, crw_κ, color=:gray)



ndata = 50
nrepetitions = 10
crw_κ = 3.4
n = 1_000
w = range(0, 1, ndata)
brw_κ = 10
nsteps = df.nsteps[1]
rvalues = df.rvalues[1]
p = [get_likelihood(nrepetitions, nsteps, brw_κ, crw_κ, w, n, rvalues) for w in w]
scatterlines(w, p)



#
#
# @showprogress for (j, grp) in enumerate(groupby(df, :id))
#     for (i, brw_κ) in enumerate(brw_κ)
#         r = optimize(w -> -get_likelihood(nrepetitions, grp.nsteps, brw_κ, crw_κ, w, n, grp.rvalues), 0, 1, Brent(); abs_tol=0.01)
#         w[j, i] = r.minimizer
#         p[j, i] = -r.minimum
#     end
# end
#
# fig = Figure()
# ax = Axis(fig[1,1])
# l = lines!(ax, brw_κ, w, color=p)
# Colorbar(fig[1,2], l)
#
#
#
# brw_κ = 5
# n = 100_000
# w = range(0.55, 0.65, 50)
# l = [get_likelihood(nrepetitions, grp.nsteps, brw_κ, crw_κ, w, n, grp.rvalues) for w in w]
# lines(w, l)
#
# r = optimize(w -> get_likelihood(nrepetitions, grp.nsteps, brw_κ, crw_κ, w, n, grp.rvalues), 0, 1)
#
# using CurveFit
#
# i = sortperm(vec(l[1]))[end - 2ndata:end]
# ind = CartesianIndices((ndata, ndata))[i]
# x = vec(brw_κ[first.(Tuple.(ind))])
# y = vec(w[last.(Tuple.(ind))])
# # M, ind = findmax(l[1], dims = 2)
# # keep = M .> 10_000
# # x = vec(brw_κ[first.(Tuple.(ind[keep]))])
# # y = vec(w[last.(Tuple.(ind[keep]))])
# fit = curve_fit(RationalPoly, x, y, 1, 2)
# heatmap(brw_κ, w, l[1], axis=(; limits=(extrema(brw_κ), (0, 1))))
# scatter!(x, y)
# x0 = range(extrema(brw_κ)..., 100)
# lines!(x0, fit.(x0), color = :red)
#
#
# itp = cubic_spline_interpolation((brw_κ, w), l[1])
#
# x = range(extrema(brw_κ)..., 100)
# function fun(p) 
#     n1, n2, d1, d2, d3 = p
#     f = RationalPoly([n1, n2], [d1, d2, d3])
#     y = f.(x)
#     keep = findall(x -> 0 < x < 1, y)
#     -sum(itp.(x[keep], y[keep]))
# end
# using Optim
# r = optimize(fun, vcat(fit.num, fit.den))
# n1, n2, d1, d2, d3 = r.minimizer
# f = RationalPoly([n1, n2], [d1, d2, d3])
# # heatmap(brw_κ, w, l[1], axis=(; limits=(extrema(brw_κ), (0, 1))))
# lines!(x, f.(x))
#
#
#
#
#
#
# l = ones(ndata , ndata)
# for grp in groupby(df, :id)
#     x = [get_likelihood(nrepetitions, grp.nsteps, brw_κ, crw_κ, w, n, grp.rvalues) for brw_κ in brw_κ, w in w]
#     l .*= x
# end
# heatmap(brw_κ, w, l)
#
#
#
# heatmap(brw_κ, w, l)
#
# l = ones(n , n)
# for grp in groupby(df, :id)
#     x = [likelihood(grp.rvalues, grp.nsteps, α, β, brw_κ, w) for brw_κ in brw_κ, w in w]
#     l .*= x#/sum(x)
# end
# heatmap(brw_κ, w, l)
#
#
# function fitBeta(nrepetitions, nsteps, brw, crw, w, n)
#     rvalues = Folds.map(i -> mean_resultant_length(nrepetitions, nsteps, brw, crw, w), 1:n)
#     d = fit(Beta, rvalues)
#     return (; α = d.α, β = d.β, ll = loglikelihood(d, rvalues))
# end
#
# ndata = 15
#
# nrepetitions = 10
# nsteps = range(10, 35, ndata)
# brw_κ = range(0.1, 100, ndata)
# crw_κ = 3.4
# w = range(0, 1, ndata)
# n = 10_000
#
# using Serialization
# data = deserialize("data")
#
# data = [fitBeta(nrepetitions, nsteps, brw_κ, crw_κ, w, n) for nsteps in nsteps, brw_κ in brw_κ, w in w]
#
# α = cubic_spline_interpolation((nsteps, brw_κ, w), getfield.(data, :α))
# β = cubic_spline_interpolation((nsteps, brw_κ, w), getfield.(data, :β))
# # α = linear_interpolation((nsteps, brw_κ, w), getfield.(data, :α))
# # β = linear_interpolation((nsteps, brw_κ, w), getfield.(data, :β))
#
# using DataFrames, CSV, StaticArrays, LinearAlgebra
# const SV = SVector{2, Float64}
#
# mean_resultant_length2(degrees) = norm(Folds.mapreduce(SV ∘ reverse ∘ sincosd, +, degrees))/length(degrees) 
# df = CSV.read("semipunctatus.csv", DataFrame)
# DataFrames.transform!(df, r"\d+" => ByRow(vcat) => :degrees, [:arena_diameter, :step_size] => ByRow((d, s) -> d/2s) => :nsteps)
# DataFrames.transform!(df, :degrees => ByRow(mean_resultant_length2) => :rvalues)
# select!(df, [:id, :nsteps, :rvalues])
# # subset!(df, :id => ByRow(!=("A.semip_09")))
# # nsteps = df.nsteps
# # rvalues = df.rvalues
#
# function likelihood(rvalues, nsteps, α, β, brw_κ::Float64, w::Float64)
#     p = 0.0
#     for (rvalue, nstep) in zip(rvalues, nsteps)
#         d = Beta(α[nstep, brw_κ, w], β[nstep, brw_κ, w])
#         p += logpdf(d, rvalue)
#     end
#     exp(p)
# end
#
#
# using Optim
#
# function optimize2rvalues(rvalues, nsteps)
#     lower = first.([brw_κ, w]) .+ 1e-3
#     upper = last.([brw_κ, w]) .- 1e-3
#     initial_x = (lower .+ upper)/2
#     inner_optimizer = GradientDescent()
#     results = optimize(x -> 1/likelihood(rvalues, nsteps, α, β, x...), lower, upper, initial_x, Fminbox(inner_optimizer))
#     return (; brw_κ = results.minimizer[1], w = results.minimizer[2])
# end
# res = combine(groupby(df, :id), [:rvalues, :nsteps] => optimize2rvalues => [:brw_κ, :w]; keepkeys=false)
#
# brw_κ, w = mean.(eachcol(res))
#
# scatter(res.brw_κ, res.w, axis=(; xlabel="Compass error (κ)", ylabel="Weight", limits=((0, 70), (0, 1))))
#
# n = 1000
# brw_κ = range(0.1, 100, n)
# w = range(0, 1, n)
# l = ones(n , n)
# for grp in groupby(df, :id)
#     x = [likelihood(grp.rvalues, grp.nsteps, α, β, brw_κ, w) for brw_κ in brw_κ, w in w]
#     l .*= x#/sum(x)
# end
# heatmap(brw_κ, w, l)
#
# fig = Figure()
# for (i, grp) in enumerate(groupby(df, :id))
#     l = [likelihood(grp.rvalues, grp.nsteps, α, β, brw_κ, w) for brw_κ in brw_κ, w in w]
#     ax = Axis(fig[i, 1])
#     heatmap!(ax, brw_κ, w, l)
# end
#
