using MAT, GLMakie

vars = matread("kappa_sweep_results_CN_230918.mat")
x = vec(vars["w_array"][Int.(vars["index2"])])
y = vec(vars["BRW_array"][Int.(vars["index1"])])

scatter(x, y, axis = (; yscale=log10, limits=((0, 1),(10^-1, 10^4))))

