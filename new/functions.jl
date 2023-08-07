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

mean_resultant_length(θs) = sqrt(mean(cos.(θs))^2 + mean(sin.(θs))^2)

function mean_resultant_length(nrepetitions, nsteps, brw::VonMises, crw::VonMises, w)
    azimuths = [get_exit_azimuth(nsteps, brw, crw, w) for j in 1:nrepetitions]
    mean_resultant_length(azimuths)
end

mean_resultant_length(nrepetitions, nsteps, brw_κ::Real, crw_κ::Real, w::Real) = mean_resultant_length(nrepetitions, nsteps, VonMises(brw_κ), VonMises(crw_κ), w)

mean_resultant_length(d::VonMises) = besselix(1, d.κ) / d.I0κx
