module BeetleModel

using Statistics, LinearAlgebra
using Distributions, StaticArrays, Folds

const SV = SVector{2, Float64}

export mean_resultant_length, get_exit_azimuth

function next_step(θ, brw_θ, crw_θ, w)
    brwyx = sincos(brw_θ)
    crwyx = sincos(θ + crw_θ)
    y, x = @. w*brwyx + (1 - w)*crwyx
    return (atan(y, x), SV(x, y))
end

function get_exit_point(nsteps, brw, crw, w)
    xy = zero(SV)
    θ = 0.0
    while norm(xy) < nsteps
        θ, Δ = next_step(θ, rand(brw), rand(crw), w)
        xy += Δ
    end
    return nsteps*normalize(xy) # shortens the last step of the beetle so it crosses the nsteps circle around the origin
end

function get_exit_azimuth(nsteps, brw, crw, w)
    xy = get_exit_point(nsteps, brw, crw, w)
    α = atand(reverse(xy)...)
    return 5round(Int, α/5)
end

mean_resultant_length(nrepetitions, nsteps, brw::VonMises, crw::VonMises, w) = norm(Folds.mapreduce(_ -> get_exit_point(nsteps, brw, crw, w), +, 1:nrepetitions; init=zero(SV)))/nrepetitions/nsteps
mean_resultant_length(nrepetitions, nsteps, brw_κ::Real, crw_κ::Real, w) = mean_resultant_length(nrepetitions, nsteps, VonMises(brw_κ), VonMises(crw_κ), w)
mean_resultant_length(nrepetitions, nsteps, brw_κ::Real, crw::VonMises, w) = mean_resultant_length(nrepetitions, nsteps, VonMises(brw_κ), crw, w)

end
