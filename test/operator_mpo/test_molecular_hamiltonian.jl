using Test
include("../../src/operator_mpo/molecular_hamiltonian.jl")

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

# ==========================================================

t = [1.0 1; 1 1]
t_symbols = to_symbols(t, "t")

oed_mpo = one_electron_diagonal(2)
matrix = to_matrix(oed_mpo, s)
matrix = matrix.subs(t_symbols)
matrix = N(matrix.tolist())

expected_matrix = zeros(16, 16)
expected_matrix += t[1, 1] * kron(s[number], s[id])
expected_matrix += t[2, 2] * kron(s[id], s[number])

@test isapprox(matrix, expected_matrix, atol = 0.001)

# ==========================================================



oe_mpos = one_electron_mpos(2)

t = rand(2,2)
matrix = to_matrix(oe_mpos, t)

expected_matrix = zeros(16, 16)
expected_matrix += t[1, 1] * kron(s[number], s[id])
expected_matrix += t[2, 2] * kron(s[id], s[number])

expected_matrix += t[2, 1] * (
    kron(s[id], s[c_up]) * kron(s[a_up], s[id]) +
    kron(s[id], s[c_down]) * kron(s[a_down], s[id])
)

expected_matrix += t[1, 2] * (
    kron(s[c_up], s[id]) * kron(s[id], s[a_up]) +
    kron(s[c_down], s[id]) * kron(s[id], s[a_down])
)

# TODO - et sans la valeur absolue ?
@test isapprox(abs.(matrix), abs.(expected_matrix), atol = 0.001)

# kron(s._c_up, s._a_up) == kron(s._zc_up_left, s._za_up_left)