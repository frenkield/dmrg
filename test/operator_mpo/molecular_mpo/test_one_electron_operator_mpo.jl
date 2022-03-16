using Test
include("../../../operator_mpo/molecular_mpo/one_electron_operator_mpo.jl")
include("../../../src/dmrg/mpo.jl")

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

oe_mpo = OneElectronOperatorMPO(2)
s = oe_mpo.molecular_system

t = rand(2,2)
matrix = to_matrix(oe_mpo, t)

expected_matrix = zeros(16, 16)
expected_matrix += t[1, 1] * kron(s._number, s._id)
expected_matrix += t[2, 2] * kron(s._id, s._number)

expected_matrix += t[2, 1] * (
    kron(s._id, s._c_up) * kron(s._a_up, s._id) +
    kron(s._id, s._c_down) * kron(s._a_down, s._id)
)

expected_matrix += t[1, 2] * (
    kron(s._c_up, s._id) * kron(s._id, s._a_up) +
    kron(s._c_down, s._id) * kron(s._id, s._a_down)
)

# TODO - et sans la valeur absolue ?
@test isapprox(abs.(matrix), abs.(expected_matrix), atol = 0.001)

# ==================================================================

oe_mpo = OneElectronOperatorMPO(2)
s = oe_mpo.molecular_system

t = rand(2, 2)
symbol_dict = to_symbols(t, "t")

q, r = qr(t)
to_symbols!(symbol_dict, collect(q), "q")
to_symbols!(symbol_dict, collect(r), "r")

tensor_mpo = MPO(oe_mpo.mpos[1], s, symbol_dict)
matrix = to_matrix(tensor_mpo)

expected_matrix = to_matrix(oe_mpo, t)

# ==================================================================

oe_mpo = OneElectronOperatorMPO(3)
t = rand(3, 3)

mpos = to_mpos(oe_mpo, t)
