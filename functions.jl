function next_step(θ, brwθ::Float64, crwθ::Float64, w)
    x = w*cos(brwθ) + (1 - w)*cos(θ + crwθ) 
    y = w*sin(brwθ) + (1 - w)*sin(θ + crwθ) 
    xy = SVector{2, Float64}(x, y)
    return (atan(y, x), normalize(xy))
end

next_step(θ, brw::VonMises, crw::VonMises, w) = next_step(θ, rand(brw), rand(crw), w)

function get_exit_azimuth(nsteps, brw, crw, w)
    xy = zero(SVector{2, Float64})
    θ = 0.0
    while norm(xy) < nsteps
        θ, Δ = next_step(θ, brw, crw, w)
        xy += Δ
    end
    return atan(reverse(xy)...)
end

function get_track(nsteps, brw, crw, w)
    n = 5nsteps
    xy = Vector{SVector{2, Float64}}(undef, n)
    xy[1] = zero(SVector{2, Float64})
    θ = 0.0
    for i in 2:n
        θ, Δ = next_step(θ, brw, crw, w)
        xy[i] = xy[i - 1] + Δ
        if  norm(xy[i]) > nsteps
            return xy[1:i-1]
        end
    end
    return xy
end

function tna(κ, r)
    Ap = besselix(1, κ)/besselix(0, κ)
    return κ - (Ap - r)/(1 - Ap^2 - 1/κ*Ap)
end

function var2κ(v, n = 2)
    @assert v < 1 "a Von Mises distribution is not defined for variances ≥ 1"
    r = 1 - v
    κ = r*(2 - r^2)/(1 - r^2)
    for i in 1:n
        κ = tna(κ, r)
    end
    return κ
end

std2κ(σ, n = 2) = var2κ(σ^2/2, n)

mean_resultant_length(θs) = sqrt(mean(cos.(θs))^2 + mean(sin.(θs))^2)

function mean_resultant_length(nrepetitions, nsteps, brw::VonMises, crw::VonMises, w)
    # azimuths = Vector{Float64}(undef, nrepetitions)
    # Threads.@threads for i in 1:nrepetitions
    #     azimuths[i] = get_exit_azimuth(nsteps, brw, crw, w)
    # end
    azimuths = [get_exit_azimuth(nsteps, brw, crw, w) for j in 1:nrepetitions]
    mean_resultant_length(azimuths)
end

mean_resultant_length(nrepetitions, nsteps, brw_σ::Truncated, crw_σ::Truncated, w_σ::Truncated) = mean_resultant_length(nrepetitions, nsteps, rand(brw_σ), rand(crw_σ), rand(w_σ))

mean_resultant_length(nrepetitions, nsteps, brw_σ::Real, crw_σ::Real, w_σ::Real) = mean_resultant_length(nrepetitions, nsteps, VonMises(std2κ(brw_σ)), VonMises(std2κ(crw_σ)), w_σ)

mean_resultant_length(d::VonMises) = besselix(1, d.κ) / d.I0κx

function Distributions.std(d::VonMises)
    R = mean_resultant_length(d)
    # sqrt(-2log(R))
    sqrt(2*(1 - R))
end

