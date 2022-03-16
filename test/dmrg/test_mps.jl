using Test
using LinearAlgebra
include("../../src/dmrg/mps.jl")

up = [1; 0]
down = [0; 1]

# ==========================================================

a = randn(2, 2)
mps = MPS(a)
@test a ≈ mps[1].tensor[:, 1, :] * mps[2].tensor[:, :, 1]'

@test a[1, 1] ≈ compute_value(mps, (1, 1))
@test a[2, 1] ≈ compute_value(mps, (2, 1))
@test a[1, 2] ≈ compute_value(mps, (1, 2))
@test a[2, 2] ≈ compute_value(mps, (2, 2))

a = randn(2, 2, 2)
mps = MPS(a)
@test a[1, 1, 1] ≈ compute_value(mps, (1, 1, 1))

a = randn(2, 2, 2, 2)
mps = MPS(a)
@test a[1, 1, 1, 1] ≈ compute_value(mps, (1, 1, 1, 1))

a = zeros(Float64, 2, 2, 2)
a[1, 1, 1] = 1
a[2, 1, 1] = 1
a[1, 2, 1] = 1
mps = MPS(a)

for c in CartesianIndices(a)
    @test isapprox(a[c.I...], compute_value(mps, c.I), atol=0.0000001)
end

# ==========================================================

a = Float64.(reshape(1:8, (2,2,2)))
mps = MPS(a)

for c in CartesianIndices(a)
    @test isapprox(a[c.I...], compute_value(mps, c.I), atol=0.0000001)
end

a = randn(2, 2, 2)
mps = MPS(a)

for c in CartesianIndices(a)
    @test isapprox(a[c.I...], compute_value(mps, c.I), atol=0.0000001)
end

# ==========================================================

a = randn(2, 2, 2)
mps = MPS(a)
for c in CartesianIndices(a)
    @test isapprox(a[c.I...], compute_value(mps, c.I), atol=0.0000001)
end

a = randn(2, 2, 2, 2)
mps = MPS(a)
for c in CartesianIndices(a)
    @test isapprox(a[c.I...], compute_value(mps, c.I), atol=0.0000001)
end

# ==========================================================

a = randn(4, 4, 4)
mps = MPS(a)
for c in CartesianIndices(a)
    @test isapprox(a[c.I...], compute_value(mps, c.I), atol=0.0000001)
end

a = randn(4, 4, 4, 4)
mps = MPS(a)
for c in CartesianIndices(a)
    @test isapprox(a[c.I...], compute_value(mps, c.I), atol=0.0000001)
end

a = randn(2, 2, 2, 2, 2)
mps = MPS(a)
for c in CartesianIndices(a)
    @test isapprox(a[c.I...], compute_value(mps, c.I), atol=0.0000001)
end

a = randn(4, 4, 4, 4, 4)
mps = MPS(a)
for c in CartesianIndices(a)
    @test isapprox(a[c.I...], compute_value(mps, c.I), atol=0.0000001)
end

a = randn(4, 4, 4, 4, 4, 4)
mps = MPS(a)
for c in CartesianIndices(a)
    @test isapprox(a[c.I...], compute_value(mps, c.I), atol=0.0000001)
end

# a = MPS(randn(4, 4))
# b = MPS(randn(4, 4))
# a_plus_b = a + b
# for c in CartesianIndices((4, 4))
#     sum_value = compute_value(a, c.I) + compute_value(b, c.I)
#     @test isapprox(compute_value(a_plus_b, c.I), sum_value, atol=0.0000001)
# end

# a = MPS(randn(2, 2, 2))
# b = MPS(randn(2, 2, 2))
# a_plus_b = a + b
# for c in CartesianIndices((2, 2, 2))
#     sum_value = compute_value(a, c.I) + compute_value(b, c.I)
#     @test isapprox(compute_value(a_plus_b, c.I), sum_value, atol=0.0000001)
# end

# a = MPS(randn(4, 4, 4, 4, 4))
# b = MPS(randn(4, 4, 4, 4, 4))
# a_plus_b = a + b
# for c in CartesianIndices((4, 4, 4, 4, 4))
#     sum_value = compute_value(a, c.I) + compute_value(b, c.I)
#     @test isapprox(compute_value(a_plus_b, c.I), sum_value, atol=0.0000001)
# end

a = randn(4, 4, 4, 4, 4)
b = randn(4, 4, 4, 4, 4)
mps_a = MPS(a)
mps_b = MPS(b)
@test dot(mps_a, mps_b) ≈ dot(a, b)

# ==========================================================

a = randn(2, 2, 2, 2)
mps = MPS(a)
for c in CartesianIndices(a)
    @test isapprox(a[c.I...], compute_value(mps, c.I), atol=0.0000001)
end

# ==========================================================

a = [rand(1:100, 2) for i in 1:4]
t = kronv(a[1], a[2], a[3], a[4])
expected = zeros(Int, 2, 2, 2, 2)

for c in CartesianIndices((2, 2, 2, 2))
    i,j,k,l = c.I
    expected[i, j, k, l] = a[1][i] * a[2][j] * a[3][k] * a[4][l]
end

@test expected == t

# ==========================================================

# TODO - on peut supprimer cette section car on a supprimé la
#        compression durant la creation du MPS

state = kronv(up, up) + kronv(down, up) + kronv(up, down) + kronv(down, down)
mps = MPS(state)
# @test bond_dimensions(mps) == [1]

state = kronv(up, up, up)
mps = MPS(state)
# @test bond_dimensions(mps) == [1, 1]

state = kronv(up, down, up)
mps = MPS(state)
# @test bond_dimensions(mps) == [1, 1]

state = kronv(up, up) + kronv(down, down)
mps = MPS(state)
@test bond_dimensions(mps) == [2]

state = kronv(up, up, up) + kronv(up, down, down)
mps = MPS(state)
# @test bond_dimensions(mps) == [1, 2]

state = kronv(up, up, up) + kronv(up, down, down)
mps = MPS(state)
# @test bond_dimensions(mps) == [1, 2]

state = kronv(up, up, up) + kronv(down, down, up)
mps = MPS(state)
# @test bond_dimensions(mps) == [2, 1]

state = kronv(up, up) + kronv(down, up) + kronv(up, down) + kronv(down, down)
mps = MPS(state)
# @test bond_dimensions(mps) == [1]

# ==========================================================

a = randn(2, 2, 2, 2) .* 10
mps = MPS(a)
@test isapprox(norm(a), norm(mps), atol=0.0000001)

# ==========================================================

matrix_1 = [0.342 0.239; 0.993 0.690]
matrix_2 = [0.360 0.644; 0.863 0.075]

# du code python
mps_matrix_1 = [-0.32617966 -0.94530779; -0.94530779  0.32617966]
mps_matrix_2 = [-8.48349112e-01 -3.40388699e-04; -9.61127094e-01 4.53939396e-04]

@test isapprox(matrix_1 * matrix_2', mps_matrix_1 * mps_matrix_2',
               atol=0.0000001)

mps = MPS(rand(2, 2))
mps[1].tensor = reshape(mps_matrix_1, (size(mps_matrix_1)[1], 1, size(mps_matrix_1)[2]))
mps[2].tensor = reshape(mps_matrix_2, (size(mps_matrix_2)..., 1))

@test isapprox(contract(mps), mps_matrix_1 * mps_matrix_2',
               atol=0.0000001)

d = svd(matrix_1)
sv = Diagonal(d.S) * d.Vt
computed_matrix_2 = sv * matrix_2'
@test isapprox(computed_matrix_2', mps_matrix_2, atol=0.0000001)

# ==========================================================

mps = MPS(rand(2, 2))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test mps[i] == mps_copy[i]
end

mps = MPS(rand(4, 4))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test mps[i] == mps_copy[i]
end

mps = MPS(rand(2, 2, 2))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test size(mps[i].matrix) == size(mps_copy[i].matrix)
end

mps = MPS(rand(4, 4, 4))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test size(mps[i].matrix) == size(mps_copy[i].matrix)
end

mps = MPS(rand(2, 2, 2, 2))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test size(mps[i].matrix) == size(mps_copy[i].matrix)
end

mps = MPS(rand(4, 4, 4, 4))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test size(mps[i].matrix) == size(mps_copy[i].matrix)
end

mps = MPS(rand(2, 2, 2, 2, 2, 2))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test size(mps[i].matrix) == size(mps_copy[i].matrix)
end
@test mps[end] == mps_copy[end]

mps = MPS(rand(4, 4, 4, 4, 4, 4))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test size(mps[i].matrix) == size(mps_copy[i].matrix)
end
@test mps[end] == mps_copy[end]

# ==========================================================

mps = MPS(rand(2, 2, 2))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test mps[i] == mps_copy[i]
end

mps = MPS(rand(4, 4, 4))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test mps[i] == mps_copy[i]
end

mps = MPS(rand(2, 2, 2, 2))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test mps[i] == mps_copy[i]
end

mps = MPS(rand(4, 4, 4, 4))
mps_copy = copy(mps)
update_matrices!(mps)
for i in 1:mps.site_count
    @test mps[i] == mps_copy[i]
end

# ==========================================================

mps = random_mps(2, 2, max_bond_dimension=4)
mps_copy = copy(mps)
left_normalize!(mps)
@test isapprox(norm(mps), norm(mps_copy), atol=0.0000001)

mps = random_mps(2, 20)
#mps = random_mps(2, 20, max_bond_dimension=20)
mps_copy = copy(mps)
# @show size(mps)
