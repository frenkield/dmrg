using Test
using LinearAlgebra
include("../../src/dmrg/mps.jl")
include("../../src/operator_mpo/heisenberg_hamiltonian.jl")

a = rand(2, 2)
a = a ./ norm(a)
mps = MPS(a)
@test isapprox(norm(mps), 1, atol=0.000001)
@test !isapprox(mps[2].tensor[:, :] * mps[2].tensor[:, :]', I(2), atol=0.00001)

left_normalize_tensor!(mps, 1)
@test isapprox(norm(mps), 1, atol=0.000001)
@test isapprox(contract(mps), a, atol=0.00001)
@test isapprox(mps[1].tensor[:, 1, :] * mps[1].tensor[:, 1, :]', I(2), atol=0.00001)
@test !isapprox(mps[2].tensor[:, :] * mps[2].tensor[:, :]', I(2), atol=0.00001)

# ==========================================================

a = rand(2, 2, 2)
a = a ./ norm(a)
mps = MPS(a)
mps_copy = copy(mps)

@test isapprox(mps[1].tensor[:, 1, :] * mps[1].tensor[:, 1, :]', I(2), atol=0.00001)

left_normalize_tensor!(mps, 1)
@test isapprox(norm(mps), 1, atol=0.000001)
@test isapprox(contract(mps), a, atol=0.00001)
@test isapprox(mps[1].tensor[:, 1, :] * mps[1].tensor[:, 1, :]', I(2), atol=0.00001)

left_normalize_tensor!(mps, 2)
@test size(mps[3]) == size(mps_copy[3])
@test size(mps[2]) == size(mps_copy[2])
@test size(mps[1]) == size(mps_copy[1])

@test isapprox(norm(mps), 1, atol=0.000001)
@test isapprox(contract(mps), a, atol=0.00001)
@test !isapprox(mps[3].tensor[:, :] * mps[3].tensor[:, :]', I(2), atol=0.00001)
@test isapprox(mps[1].tensor[:, 1, :] * mps[1].tensor[:, 1, :]', I(2), atol=0.00001)

# ==========================================================

a = rand(2, 2)
mps = MPS(a)
left_normalize_tensor!(mps, 1)
@test isapprox(contract(mps), a, atol=0.00001)

a = rand(2, 2, 2)
mps = MPS(a)
left_normalize_tensor!(mps, 1)
@test isapprox(contract(mps), a, atol=0.00001)
left_normalize_tensor!(mps, 2)
@test isapprox(contract(mps), a, atol=0.00001)

a = rand(2, 2, 2, 2)
mps = MPS(a)
left_normalize_tensor!(mps, 1)
@test isapprox(contract(mps), a, atol=0.00001)
left_normalize_tensor!(mps, 2)
@test isapprox(contract(mps), a, atol=0.00001)
left_normalize_tensor!(mps, 3)
@test isapprox(contract(mps), a, atol=0.00001)

a = rand(4, 4, 4, 4)
mps = MPS(a)
left_normalize_tensor!(mps, 1)
@test isapprox(contract(mps), a, atol=0.00001)
left_normalize_tensor!(mps, 2)
@test isapprox(contract(mps), a, atol=0.00001)
left_normalize_tensor!(mps, 3)
@test isapprox(contract(mps), a, atol=0.00001)

# ==========================================================

a = rand(2, 2)
a = a ./ norm(a)
mps = MPS(a)
@test isapprox(mps[1].tensor[:, 1, :] * mps[1].tensor[:, 1, :]', I(2), atol=0.00001)
@test !isapprox(mps[2].tensor[:, :] * mps[2].tensor[:, :]', I(2), atol=0.00001)

right_normalize_tensor!(mps, 2)
@test isapprox(norm(mps), 1, atol=0.000001)
@test isapprox(contract(mps), a, atol=0.00001)
@test isapprox(mps[2].tensor[:, :] * mps[2].tensor[:, :]', I(2), atol=0.00001)
@test !isapprox(mps[1].tensor[:, 1, :] * mps[1].tensor[:, 1, :]', I(2), atol=0.00001)

a = rand(2, 2, 2)
a = a ./ norm(a)
mps = MPS(a)
@test isapprox(norm(mps), 1, atol=0.000001)
@test !isapprox(mps[3].tensor[:, :] * mps[3].tensor[:, :]', I(2), atol=0.00001)

mps_copy = copy(mps)

right_normalize_tensor!(mps, 3)
@test isapprox(norm(mps), 1, atol=0.000001)
@test isapprox(contract(mps), a, atol=0.00001)

right_normalize_tensor!(mps, 2)
@test isapprox(norm(mps), 1, atol=0.000001)
@test isapprox(contract(mps), a, atol=0.00001)

@test size(mps[3]) == size(mps_copy[3])
@test size(mps[2]) == size(mps_copy[2])
@test size(mps[1]) == size(mps_copy[1])

@test isapprox(mps[3].tensor[:, :] * mps[3].tensor[:, :]', I(2), atol=0.00001)
@test !isapprox(mps[1].tensor[:, 1, :] * mps[1].tensor[:, 1, :]', I(2), atol=0.00001)

# ========================================================

a = rand(2, 2)
mps = MPS(a)
mps_copy = copy(mps)

right_normalize!(mps)
@test isapprox(contract(mps), a, atol=0.00001)
@test mps[1] != mps_copy[1]
@test mps[2] != mps_copy[2]

left_normalize!(mps)
@test isapprox(contract(mps), a, atol=0.00001)

@test isapprox(mps[1].tensor, mps_copy[1].tensor, atol=0.00001)
@test mps[1] == mps_copy[1]
@test mps[2] == mps_copy[2]

# ========================================================

h_2 = HeisenbergHamiltonian(2)
mpo = h_2.tensor_mpo

a = rand(2, 2)
mps = MPS(a)
mps_copy = copy(mps)
energy = dot(mps, mpo * mps)

for i in 1:5

    right_normalize!(mps)
    @test isapprox(dot(mps, mpo * mps), energy, atol=0.00001)
    @test isapprox(mps, mps_copy)
    @test !isidentical(mps, mps_copy)

    left_normalize!(mps)
    @test isapprox(dot(mps, mpo * mps), energy, atol=0.00001)
    @test mps == mps_copy
    @test isidentical(mps, mps_copy)

end

# ========================================================

h_3 = HeisenbergHamiltonian(3)
mpo = h_3.tensor_mpo

a = rand(2, 2, 2)
mps = MPS(a)
mps_copy = copy(mps)
energy = dot(mps, mpo * mps)

for i in 1:10

    right_normalize!(mps)
    @test isapprox(dot(mps, mpo * mps), energy, atol=0.00001)
    @test isapprox(mps, mps_copy)
    @test !isidentical(mps, mps_copy)

    left_normalize!(mps)
    @test isapprox(dot(mps, mpo * mps), energy, atol=0.00001)
    @test isidentical(mps, mps_copy)

end

# ========================================================

h_4 = HeisenbergHamiltonian(4)
mpo = h_4.tensor_mpo

a = rand(2, 2, 2, 2)
mps = MPS(a)
mps_copy = copy(mps)
energy = dot(mps, mpo * mps)

for i in 1:50

    right_normalize!(mps)
    @test isapprox(dot(mps, mpo * mps), energy, atol=0.00001)
    @test isapprox(mps, mps_copy)
    @test !isidentical(mps, mps_copy)

    left_normalize!(mps)
    @test isapprox(dot(mps, mpo * mps), energy, atol=0.00001)
    @test isapprox(mps, mps_copy)

#    TODO - est-ce necessaire ?
#    if i > 1
#        @test isidentical(mps, mps_copy)
#    end

end

# ========================================================

a = rand(2, 2, 2, 2)
mps = MPS(a, max_bond_dimension=2)
left_normalize!(mps)

for tensor in mps.tensors
    @test tensor.left_bond_dimension <= mps.max_bond_dimension
    @test tensor.right_bond_dimension <= mps.max_bond_dimension
end

a = rand(2, 2, 2, 2)
mps = MPS(a, max_bond_dimension=2)
right_normalize!(mps)

for tensor in mps.tensors
    @test tensor.left_bond_dimension <= mps.max_bond_dimension
    @test tensor.right_bond_dimension <= mps.max_bond_dimension
end

a = rand(2, 2, 2, 2)
mps = MPS(a, max_bond_dimension=2)
left_normalize!(mps)

a = contract(mps)
mps = MPS(a)
mps_copy = copy(mps)
mps.max_bond_dimension = 2
left_normalize!(mps)
@test isapprox(norm(mps), norm(mps_copy), atol=0.00001)

for tensor in mps.tensors
    @test tensor.left_bond_dimension <= 2
    @test tensor.right_bond_dimension <= 2
end

# ========================================================

a = rand(4, 4, 4, 4)
mps = MPS(a, max_bond_dimension=4)
left_normalize!(mps)

for tensor in mps.tensors
    @test tensor.left_bond_dimension <= mps.max_bond_dimension
    @test tensor.right_bond_dimension <= mps.max_bond_dimension
end

a = rand(4, 4, 4, 4)
mps = MPS(a, max_bond_dimension = 4)
right_normalize!(mps)

for tensor in mps.tensors
    @test tensor.left_bond_dimension <= mps.max_bond_dimension
    @test tensor.right_bond_dimension <= mps.max_bond_dimension
end

a = rand(4, 4, 4, 4)
mps = MPS(a)
mps.max_bond_dimension = 8
left_normalize!(mps)

for tensor in mps.tensors
    @test tensor.left_bond_dimension <= mps.max_bond_dimension
    @test tensor.right_bond_dimension <= mps.max_bond_dimension
end

a = rand(4, 4, 4, 4)
mps = MPS(a, max_bond_dimension=8)
right_normalize!(mps)

for tensor in mps.tensors
    @test tensor.left_bond_dimension <= 8
    @test tensor.right_bond_dimension <= 8
end

# ========================================================

a = rand(4, 4, 4, 4)
mps = MPS(a, max_bond_dimension=8)
left_normalize!(mps)

a = contract(mps)
mps = MPS(a, max_bond_dimension=8)
mps_copy = copy(mps)
left_normalize!(mps)
@test isapprox(norm(mps), norm(mps_copy), atol=0.00001)
