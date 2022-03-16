using Test
using Plots
include("../../experiments/stong_basis.jl")
include("../../experiments/units.jl")

# https://www.lct.jussieu.fr/pagesperso/chaquin/Bases.pdf
# http://www.phys.ubbcluj.ro/~vasile.chis/cursuri/cspm/course6.pdf

x_min = [-10, -10, -10]
x_max = [10, 10, 10]
basis = STO3GHydrogen()

x_3d = [[y, 0.0, 0.0] for y in 0:0.1:4]
r = [z[1] for z in x_3d]

default(linewidth=3.0)

plot(x, slater_1s.(x_3d), label="slater", linestyle=:dot)
#plot!(x, sto1g.(x_3d), label="sto1g")
#plot!(x, sto2g.(x_3d), label="sto2g")
#plot!(x, sto3g.(x_3d), label="sto3g")
display(plot!(x, sto3g_scaled.(r), label="sto3g scaled"))
display(plot!(x, sto3g_not_scaled.(r), label="sto3g not scaled"))
