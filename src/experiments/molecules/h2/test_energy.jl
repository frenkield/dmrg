using Test
using HDF5
import Arpack
include("../molecular_hamiltonian_matrix.jl")

nuclear_repulsion_energy = 1 / (0.74 / 0.529177210903)
@test isapprox(nuclear_repulsion_energy, 0.71510433908108118, atol=0.000000001)

t_qp,u_qp = read_electron_integral_tensors("data/h2/h2.ezfio.FCIDUMP")
m_qp = molecular_hamiltonian_matrix(t_qp, u_qp)
m_qp_full = m_qp + (I(16) * 0.71510433908108118)

eigenvalues,eigenvectors = Arpack.eigs(m_qp_full)
lowest_energy = -1.1372838344894252
@test isapprox(eigenvalues[1], lowest_energy, atol=0.0000001)

vac = [1; 0; 0; 0]
up = [0; 1; 0; 0]
down = [0; 0; 1; 0]
both = [0; 0; 0; 1]

quantum_package_energy = -1.1167593073974036
state_up_down_1 = kron(both, vac)
energy_up_down_1 = state_up_down_1' * m_qp_full * state_up_down_1
@test energy_up_down_1 ≈ quantum_package_energy

state_up_down_2 = kron(vac, both)
energy_up_down_2 = state_up_down_2' * m_qp_full * state_up_down_2

# pourquoi l'energie est differente ????????????
@test energy_up_down_2 ≈ 0.4626181460260801

ground_state = eigenvectors[:, 1]
ground_state_energy = ground_state' * m_qp_full * ground_state
@test isapprox(ground_state_energy, lowest_energy, atol=0.0000001)

expected_ground_state = -0.1125438868930182 * kron(vac, both) +
                               0.9936467548998544 * kron(both, vac)

@test isapprox(ground_state, expected_ground_state, atol=0.0001)



