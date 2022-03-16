using Test
include("../src/dmrg/itensor_utils.jl")
include("../src/dmrg/fcidump.jl")

# ground state of a water molecule using integral data from Quantum Package

h,v = read_electron_integral_tensors("data/h2o/h2o.ezfio.FCIDUMP")

energy,state = molecular_dmrg(h, v, max_bond_dimension=150)

nuclear_repulsion_energy = 9.168193300755693
total_energy = energy + nuclear_repulsion_energy

@test isapprox(total_energy, -75.01053577546307, atol=0.01)
