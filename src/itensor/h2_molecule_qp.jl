using Test
include("../src/dmrg/itensor_utils.jl")
include("../src/dmrg/fcidump.jl")

# ground state of a water molecule using integral data from Quantum Package

h,v = read_electron_integral_tensors("data/h2/h2.ezfio.FCIDUMP")

energy,state = molecular_dmrg(h, v)

nuclear_repulsion_energy = 0.71510433908108118
total_energy = energy + nuclear_repulsion_energy

@test isapprox(total_energy, -1.1372838344894254, atol=0.00000001)
