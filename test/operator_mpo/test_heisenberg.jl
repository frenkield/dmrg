using Test
using LinearAlgebra

include("../../src/dmrg/mps.jl")
include("../../src/dmrg/mpo.jl")
include("../../src/operator_mpo/operator_mpo.jl")
include("../../src/operator_mpo/spin_system.jl")

#==
 = These DMRG tests specifically use OperatorMPO to generate MPOs.
 = The OperatorMPO is an implementation of the MPO construction method
 = described in Snajberk.
 =
 = In dmrg/test_dmrg.jl we instead use ITensor to construct MPOs.
 = Needless to say, the ITensor method is vastly superior.
 =
 = Long story short, all the OperatorMPO code (including this test) isn't particularly
 = useful.
 =#

s = SpinSystem()
@unpack id, spin_z, spin_plus, spin_minus = s
null = s.zero

h_2 = OperatorMPO([spin_z 1//2 * spin_plus 1//2 * spin_minus], [spin_z; spin_minus; spin_plus])
h_2_matrix = Float64.(N(to_matrix(h_2, s).tolist()))

expected_h_2_matrix = kron(s._spin_z, s._spin_z) + 1/2 * (kron(s._spin_plus, s._spin_minus) +
                      kron(s._spin_minus, s._spin_plus))

@test h_2_matrix == expected_h_2_matrix


h_2 = OperatorMPO([null  1//2 * spin_minus  1//2 * spin_plus  spin_z  id],
                  [id; spin_plus; spin_minus; spin_z; null])

h_2_matrix = Float64.(N(to_matrix(h_2, s).tolist()))
@test h_2_matrix == expected_h_2_matrix

vals,vecs = eigen(h_2_matrix)
@test isapprox(vals[1], -0.75, atol=0.0000001)

energy = dot(h_2_matrix * [0; 1; -1; 0], [0; 1; -1; 0]) / 2
@test isapprox(energy, -0.75, atol=0.0000001)

energy = dot(h_2_matrix * [0; -1; 1; 0], [0; -1; 1; 0]) / 2
@test isapprox(energy, -0.75, atol=0.0000001)

# =================================================================

h_3 = OperatorMPO([null  1//2 * spin_minus  1//2 * spin_plus  spin_z  id],
                  [id null null null null;
                   spin_plus null null null null;
                   spin_minus null null null null;
                   spin_z null null null null
                   null 1//2*spin_minus 1//2*spin_plus spin_z id],
                  [id; spin_plus; spin_minus; spin_z; null])

h_3_matrix = Float64.(N(to_matrix(h_3, s).tolist()))

expected_h_3_matrix = kron(s._spin_z, s._spin_z, I(2)) +
                      1/2 * (kron(s._spin_plus, s._spin_minus, I(2)) +
                      kron(s._spin_minus, s._spin_plus, I(2))) +
                      kron(I(2), s._spin_z, s._spin_z) +
                      1/2 * (kron(I(2), s._spin_plus, s._spin_minus) +
                      kron(I(2), s._spin_minus, s._spin_plus))

@test h_3_matrix == expected_h_3_matrix

# =================================================================

h_4 = OperatorMPO([null  1//2 * spin_minus  1//2 * spin_plus  spin_z  id],
                  [id null null null null;
                   spin_plus null null null null;
                   spin_minus null null null null;
                   spin_z null null null null
                   null 1//2*spin_minus 1//2*spin_plus spin_z id],
                  [id null null null null;
                   spin_plus null null null null;
                   spin_minus null null null null;
                   spin_z null null null null
                   null 1//2*spin_minus 1//2*spin_plus spin_z id],
                  [id; spin_plus; spin_minus; spin_z; null])

h_4_matrix = Float64.(N(to_matrix(h_4, s).tolist()))

expected_h_4_matrix =
    kron(s._spin_z, s._spin_z, I(2), I(2)) +
    1/2 * (kron(s._spin_plus, s._spin_minus, I(2), I(2)) +
    kron(s._spin_minus, s._spin_plus, I(2), I(2))) +
    kron(I(2), s._spin_z, s._spin_z, I(2)) +
    1/2 * (kron(I(2), s._spin_plus, s._spin_minus, I(2)) +
    kron(I(2), s._spin_minus, s._spin_plus, I(2))) +
    kron(I(2), I(2), s._spin_z, s._spin_z) +
    1/2 * (kron(I(2), I(2), s._spin_plus, s._spin_minus) +
    kron(I(2), I(2), s._spin_minus, s._spin_plus))

@test h_4_matrix == expected_h_4_matrix

# =================================================================

h_2_operator_mpo = OperatorMPO([null  1//2 * spin_minus  1//2 * spin_plus  spin_z  id],
                  [id; spin_plus; spin_minus; spin_z; null])

h_2_tensor_mpo = MPO(h_2_operator_mpo, s)

singlet_tensor = 1 / sqrt(2) * (kronv(s._up, s._down) - kronv(s._down, s._up))
singlet_mps = MPS(singlet_tensor)

energy = dot(singlet_mps, h_2_tensor_mpo * singlet_mps)
@test isapprox(energy, -0.75, atol=0.0000001)

# =================================================================

h_3_operator_mpo = OperatorMPO([null  1//2 * spin_minus  1//2 * spin_plus  spin_z  id],
                               [id null null null null;
                               spin_plus null null null null;
                               spin_minus null null null null;
                               spin_z null null null null
                               null 1//2*spin_minus 1//2*spin_plus spin_z id],
                               [id; spin_plus; spin_minus; spin_z; null])

h_3_tensor_mpo = MPO(h_3_operator_mpo, s)
h_3_matrix = to_float64_matrix(h_3_operator_mpo, s)

expected_h_3_matrix = kron(s._spin_z, s._spin_z, I(2)) +
                      1/2 * (kron(s._spin_plus, s._spin_minus, I(2)) +
                      kron(s._spin_minus, s._spin_plus, I(2))) +
                      kron(I(2), s._spin_z, s._spin_z) +
                      1/2 * (kron(I(2), s._spin_plus, s._spin_minus) +
                      kron(I(2), s._spin_minus, s._spin_plus))

state_tensor = rand(2, 2, 2)
state_vector = reshape(state_tensor, (8, 1))
state_mps = MPS(state_tensor)

expected_energy = state_vector' * expected_h_3_matrix * state_vector
energy = dot(state_mps, h_3_tensor_mpo * state_mps)

# @show expected_energy, energy

# @test isapprox(energy, expected_energy, atol=0.0000001)
