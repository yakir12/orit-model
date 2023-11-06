using Statistics, LinearAlgebra
using Distributions, StaticArrays, Folds

const SV = SVector{2, Float64}

function next_step(θ, brw_θ, crw_θ, w)
    brwyx = sincos(brw_θ)
    crwyx = sincos(θ + crw_θ)
    y, x = @. w*brwyx + (1 - w)*crwyx
    return (atan(y, x), SV(x, y))
end

function create_track(nsteps, brw, crw, w)
    n = 10nsteps
    xys = Vector{SV}(undef, n)
    count = 1
    xys[count] = zero(SV)
    θ = 0.0
    while norm(xys[count]) < nsteps
        θ, Δ = next_step(θ, rand(brw), rand(crw), w)
        xys[count + 1] = xys[count] + Δ
        count += 1
        if count > n
            break
        end
    end
    if count ≤ n
        deleteat!(xys, count:n)
    end
    xys[end] =  nsteps*normalize(xys[end]) # shortens the last step of the beetle so it crosses the nsteps circle around the origin
    return xys
end

using GLMakie, Folds, Distributions, Interpolations, KernelDensity, Optim

nsteps = 1_000_000
brw_κ = 6
brw = VonMises(brw_κ)
crw_κ = 4
crw = VonMises(crw_κ)
w = 0.5
xys = create_track(nsteps, brw, crw, w)

f = hist(norm.(diff(xys)), bins=100; axis=(;xlabel="Step length", ylabel="Counts", limits=((0,1), nothing)))
save("step_lengths.png", f.figure)


fig = Figure()
ax = Axis(fig[1,1],aspect=DataAspect()) 
lines!(ax, Circle(zero(Point2f), nsteps))
lines!(ax, xys)

θs = [atan(reverse(Δ)...) for Δ in diff(xys)]
xys2 = Vector{SV}(undef, length(xys))
xys[1] = zero(SV)
θ = 0.0
for (i, θ) in enumerate(θs)
    xys2[i + 1] = xys2[i] + SV(reverse(sincos(θ)))
end

lines!(ax, xys2)
