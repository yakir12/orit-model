using CSV, DataFrames
using Statistics, LinearAlgebra
using Distributions, StaticArrays
using GLMakie, Folds

const SV = SVector{2, Float64}

mean_resultant_length(degrees) = norm(Folds.mapreduce(SV ∘ reverse ∘ sincosd, +, degrees)/length(degrees)) 

df = CSV.read("semipunctatus.csv", DataFrame)
transform!(df, r"\d+" => ByRow(vcat) => :degrees, [:arena_diameter, :step_size] => ByRow((d, s) -> d/2s) => :nsteps)
transform!(df, :degrees => ByRow(mean_resultant_length) => :r)

fig = Figure()
ax = Axis(fig[1,1], xlabel = "Steps (#)", ylabel = "Mean resultant length")
for grp in groupby(df, :id)
    sort!(grp, :nsteps)
    color = grp.r[1] < grp.r[2] ? :black : :red
    lines!(ax, grp.nsteps, grp.r; color)
end
save("semipunctatus.png", fig)


cmp = combine(groupby(df, :id)) do grp
    sort!(grp, :nsteps)
    (; Δ = only(diff(grp.r)))
end

mean(cmp.Δ)



