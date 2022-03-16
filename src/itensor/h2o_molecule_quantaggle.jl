using Test
using HDF5
include("../src/dmrg/itensor_utils.jl")

# https://github.com/qulacs/Quantaggle_dataset/tree/master/datasets/Small_Molecules_1

io = h5open("data/h2o/H2O_sto-3g_singlet_0.96_104.5deg_0.96.hdf5", "r")
data = read(io)
h = data["one_body_integrals"]
v = data["two_body_integrals"]

energy,state = molecular_dmrg(h, v)

nuclear_repulsion_energy = 9.168193300755693
total_energy = energy + nuclear_repulsion_energy

@test isapprox(total_energy, -75.01315109061886, atol=0.0001)
