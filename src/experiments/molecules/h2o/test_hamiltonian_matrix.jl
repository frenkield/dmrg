using Test
using HDF5
using Arpack
include("../molecular_hamiltonian_matrix.jl")

io = h5open("data/h2o/H2O_sto-3g_singlet_0.96_104.5deg_0.96.hdf5", "r")
data = read(io)

fci_energy = data["fci_energy"]
nuclear_repulsion = data["nuclear_repulsion"]

h,v = read_electron_integral_tensors("data/h2o/h2o.ezfio.FCIDUMP")

m_h2o = molecular_hamiltonian_matrix(h, v)

println("adding nuclear repulsion")
m_h2o += sparse(I(4^7) * nuclear_repulsion)

decomposition = eigs(m_h2o)
