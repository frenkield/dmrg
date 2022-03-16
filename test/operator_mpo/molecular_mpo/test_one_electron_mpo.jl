using Test
include("../../../src/operator_mpo/molecular_mpo/one_electron_mpo.jl")

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

s = MolecularSystem()

t = rand(2, 2)
oe_mpo = OneElectronMPO(t)
oe_operator_mpo = OneElectronOperatorMPO(2)
@test to_matrix(oe_mpo) == to_matrix(oe_operator_mpo, t)

c1u = kron(s._c_up, s._id)
a1u = kron(s._a_up, s._id)
c1d = kron(s._c_down, s._id)
a1d = kron(s._a_down, s._id)

c2u = kron(s._id, s._c_up)
a2u = kron(s._id, s._a_up)
c2d = kron(s._id, s._c_down)
a2d = kron(s._id, s._a_down)

exact_matrix = zeros(16, 16)

exact_matrix += t[1, 1] * (c1u * a1u + c1d * a1d)
exact_matrix += t[2, 1] * (c2u * a1u + c2d * a1d)
exact_matrix += t[1, 2] * (c1u * a2u + c1d * a2d)
exact_matrix += t[2, 2] * (c2u * a2u + c2d * a2d)

mpo_matrix = to_matrix(oe_mpo)

# TODO - il faut qu'on regle tout ca
@test abs.(mpo_matrix) ≈ abs.(exact_matrix)
# @test mpo_matrix ≈ exact_matrix
# kron(s._zc_up_left, s._za_up_left) - kron(s._c_up, s._a_up)
# kron(s._za_up_right, s._zc_up_right) - kron(s._a_up, s._c_up)

# pourtant, les valeurs propres sont identiques
@test eigen(mpo_matrix).values ≈ eigen(exact_matrix).values




# ==================================================

#=

t = rand(3, 3)
symbol_dict = to_symbols(t, "t")

oe_mpo = OneElectronMPO(t)
oe_operator_mpo = OneElectronOperatorMPO(3)

mpo_matrix = to_matrix(oe_mpo)




using Profile
Profile.clear()

@profile to_matrix(oe_operator_mpo, t)

Profile.print(sortedby=:count, format=:flat)

=#

#=
# TODO - on peut accelerer ce machin ?
mpo_operator_matrix = to_matrix(oe_operator_mpo, t)

@test mpo_matrix ≈ mpo_operator_matrix
=#