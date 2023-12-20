using Statistics, LinearAlgebra, Distributions, Folds

function next_step(θ, brwθ, crwθ, w)
    brwyx = sincos(brwθ)
    crwyx = sincos(θ + crwθ)
    y, x = @. w*brwyx + (1 - w)*crwyx
    θ = atan(y, x)
    step = SVector{2, Float64}(x, y)
    Δ = normalize(step) # normalize the step to 1
    return (θ, Δ)
end

function get_exit_point(nsteps, brw, crw, w)
    xy = zero(SVector{2, Float64})
    θ = 0.0
    while norm(xy) < nsteps # by definition, this loop will exit when the norm of xy is larger than nsteps
        θ, Δ = next_step(θ, rand(brw), rand(crw), w)
        xy += Δ
    end
    return nsteps*normalize(xy) # so we need to shorten the last step of the beetle so it intersects with the nsteps circle around the origin
end

mean_resultant_length(nrepetitions, nsteps, brw, crw, w) = norm(Folds.mapreduce(_ -> get_exit_point(nsteps, brw, crw, w), +, 1:nrepetitions)/nrepetitions/nsteps) 

