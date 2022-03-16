using Test
using Plots
include("../../../../experiments/molecules/h2/basis.jl")

r = collect(LinRange(-4, 4, 150))

default(linewidth=3.0)

# plot(r, atomic_orbital_1_scaled.(r), label="atomic orbital 1")
# plot!(r, atomic_orbital_1_not_scaled.(r), label="atomic orbital 1 (ns)")

# plot!(r, atomic_orbital_2_scaled.(r), label="atomic orbital 2")
# plot!(r, atomic_orbital_2_not_scaled.(r), label="atomic orbital 2 (ns)")



plot(r, molecular_orbital_1_not_scaled.(r), label="molecular orbital 1")
plot!(r, molecular_orbital_2_not_scaled.(r), label="molecular orbital 2")
