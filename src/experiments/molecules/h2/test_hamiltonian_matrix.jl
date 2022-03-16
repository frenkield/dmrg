using Test
using HDF5
include("../molecular_hamiltonian_matrix.jl")

# https://github.com/qulacs/Quantaggle_dataset/tree/master/datasets/Small_Molecules_1

t_qp,u_qp = read_electron_integral_tensors("data/h2/h2.ezfio.FCIDUMP")
t_qp = round.(t_qp, digits=6)
u_qp = round.(u_qp, digits=6)

io = h5open("data/h2/H2_line_sto-3g_singlet_0.74.hdf5", "r")
data = read(io)
t_qat = round.(data["one_body_integrals"], digits=6)
u_qat = round.(data["two_body_integrals"], digits=6)

@test isapprox(t_qp, t_qat, atol=0.0000001)
@test isapprox(u_qp, u_qat, atol=0.0000001)

m_qp = molecular_hamiltonian_matrix(t_qp, u_qp)
m_qat = molecular_hamiltonian_matrix(t_qat, u_qat)

m_qp_full = m_qp + (I(16) * 0.71510433908108118)
m_qat_full = m_qat + (I(16) * 0.71510433908108118)

@test isapprox(m_qp_full, m_qat_full, atol=0.0000001)

# from openfermion
expected_matrix = [
    0.71510434 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0.24003549 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0.24003549 0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0.46261815 0 0 0 0 0 0 0 0 0.18121046 0 0 0;
    0 0 0 0 -0.53820545 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 -0.53077336 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 -0.3495629 0 0 -0.18121046 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0.3555207 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 -0.53820545 0 0 0 0 0 0 0;
    0 0 0 0 0 0 -0.18121046 0 0 -0.3495629 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 -0.53077336 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0.3555207 0 0 0 0;
    0 0 0 0.18121046 0 0 0 0 0 0 0 0 -1.11675931 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 -0.44561582 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.44561582 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.92317918
]
 
@test isapprox(m_qp_full, expected_matrix, atol=0.00001)
 




