using Test
using LinearAlgebra
include("../../src/dmrg/mps_tensor.jl")
include("../../src/dmrg/mps.jl")

a = rand(2, 2)
mps = MPS(a)

d = svd(a)

t = MPSTensor(d.U, 2, 1)
@test t.tensor == mps[1].tensor

t = MPSTensor(Diagonal(d.S) * d.Vt, 2, 2)
@test t.tensor == mps[2].tensor

# =================================================


#=

a = rand(2, 2)
mps = MPS(a)
m = tensor_to_matrix(mps, mps[1])
t = matrix_to_tensor(mps, m, size(mps[1])[2])
@test t == mps[1]

m = tensor_to_matrix(mps, mps[2])
t = matrix_to_tensor(mps, m, size(mps[2])[2])
@test t == mps[2]

let
    a = rand(2, 2, 2)
    mps = MPS(a)
    for i in 1:3
        m = tensor_to_matrix(mps, mps[i])
        t = matrix_to_tensor(mps, m, size(mps[i])[2])
        @test t == mps[i]
    end
end

let
    a = rand(2, 2, 2, 2)
    mps = MPS(a)
    for i in 1:3
        m = tensor_to_matrix(mps, mps[i])
        t = matrix_to_tensor(mps, m, size(mps[i])[2])
        @test t == mps[i]
    end
end

let
    a = rand(4, 4, 4, 4)
    mps = MPS(a)
    for i in 1:4
        m = tensor_to_matrix(mps, mps[i])
        t = matrix_to_tensor(mps, m, size(mps[i])[2])
        @test t == mps[i]
    end
end

=#