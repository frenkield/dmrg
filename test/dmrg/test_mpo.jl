using Test
include("../../src/dmrg/dmrg.jl")
include("../../src/dmrg/itensor_utils.jl")

h,v = read_electron_integral_tensors("data/h2/h2.ezfio.FCIDUMP")
sites = ITensors.siteinds("Electron", 2)
itensor_mpo = create_one_electron_mpo(sites, h)
mpo = MPO(itensor_mpo)

# =================================================================

s = SpinSystem()
@unpack id, spin_z, spin_x, spin_y, spin_plus, spin_minus, number = SpinSystem()

# =================================================================

states = [s._up, s._down]

operator_mpo = OperatorMPO(2)
add_operator_matrix!(operator_mpo, [randn() * spin_z spin_x])
add_operator_matrix!(operator_mpo, [number; randn() * id])
mpo = MPO(operator_mpo, s)
mpo_matrix = to_matrix(mpo)

for c in CartesianIndices(zeros(2, 2, 2, 2))
    i,j,k,l = c.I
    @test kron(states[j], states[l])' * mpo_matrix * kron(states[i], states[k]) ==
          compute_tensor_value(mpo, (i, j, k, l))
end

operator_mpo = OperatorMPO(2)
add_operator_matrix!(operator_mpo, [randn() * spin_z randn() * spin_x])
add_operator_matrix!(operator_mpo, [number; randn() * spin_z])
mpo = MPO(operator_mpo, s)
mpo_matrix = to_matrix(mpo)

for c in CartesianIndices(zeros(2, 2, 2, 2))
    i,j,k,l = c.I
    @test kron(states[j], states[l])' * mpo_matrix * kron(states[i], states[k]) ==
          compute_tensor_value(mpo, (i, j, k, l))
end

# =============================================================

operator_mpo = OperatorMPO(2)
add_operator_matrix!(operator_mpo, [id id])
add_operator_matrix!(operator_mpo, [id; id])
mpo = MPO(operator_mpo, s)
product_mpo = mpo * mpo
@test to_matrix(product_mpo) == to_matrix(mpo)^2

operator_mpo = OperatorMPO(2)
add_operator_matrix!(operator_mpo, [2id id])
add_operator_matrix!(operator_mpo, [id; 12id])
mpo = MPO(operator_mpo, s)
product_mpo = mpo * mpo
@test to_matrix(product_mpo) == to_matrix(mpo)^2

operator_mpo = OperatorMPO(2)
add_operator_matrix!(operator_mpo, [2spin_plus spin_z])
add_operator_matrix!(operator_mpo, [spin_minus; 2spin_z])
mpo = MPO(operator_mpo, s)
product_mpo = mpo * mpo
@test to_matrix(product_mpo) == to_matrix(mpo)^2

# ==============================================================

operator_mpo = OperatorMPO(3)
add_operator_matrix!(operator_mpo, [id id])
add_operator_matrix!(operator_mpo, [id id; id id])
add_operator_matrix!(operator_mpo, [id; id])
mpo = MPO(operator_mpo, s)
product_mpo = mpo * mpo
@test to_matrix(product_mpo) == to_matrix(mpo)^2

operator_mpo = OperatorMPO(3)
add_operator_matrix!(operator_mpo, [spin_z spin_plus])
add_operator_matrix!(operator_mpo, [spin_minus id; spin_z spin_z])
add_operator_matrix!(operator_mpo, [id; spin_plus])
mpo = MPO(operator_mpo, s)
product_mpo = mpo * mpo
@test to_matrix(product_mpo) == to_matrix(mpo)^2

operator_mpo = OperatorMPO(3)
add_operator_matrix!(operator_mpo, [2spin_z 3spin_plus])
add_operator_matrix!(operator_mpo, [4spin_minus 5id; 6spin_z 7spin_z])
add_operator_matrix!(operator_mpo, [8id; 9spin_plus])
mpo = MPO(operator_mpo, s)
product_mpo = mpo * mpo
@test to_matrix(product_mpo) == to_matrix(mpo)^2

# =============================================================

operator_mpo = OperatorMPO(2)
add_operator_matrix!(operator_mpo, [id id])
add_operator_matrix!(operator_mpo, [id; id])
mpo = MPO(operator_mpo, s)
sum_mpo = mpo + mpo
@test to_matrix(sum_mpo) == to_matrix(mpo) * 2

operator_mpo1 = OperatorMPO(3)
add_operator_matrix!(operator_mpo1, [id id])
add_operator_matrix!(operator_mpo1, [4spin_minus 5id; 6spin_z 7spin_z])
add_operator_matrix!(operator_mpo1, [id; id])
mpo1 = MPO(operator_mpo1, s)

operator_mpo2 = OperatorMPO(3)
add_operator_matrix!(operator_mpo2, [id spin_minus])
add_operator_matrix!(operator_mpo2, [id 5id; 6spin_z 7spin_z])
add_operator_matrix!(operator_mpo2, [spin_z; 5spin_z])
mpo2 = MPO(operator_mpo2, s)

sum_mpo = mpo1 + mpo2
@test to_matrix(sum_mpo) == to_matrix(mpo1) + to_matrix(mpo2)

# =============================================================

operator_mpo = OperatorMPO(2)
add_operator_matrix!(operator_mpo, [id id])
add_operator_matrix!(operator_mpo, [id; id])
mpo = MPO(operator_mpo, s)
sum_mpo = mpo + mpo
@test to_matrix(sum_mpo) == to_matrix(mpo) * 2

a = rand(2, 2)
mps = MPS(a)
mps_product = mpo * mps
a_product = contract(mps_product)

# =============================================================

operator_mpo = OperatorMPO(2)
add_operator_matrix!(operator_mpo, [id id])
add_operator_matrix!(operator_mpo, [id; id])
mpo = MPO(operator_mpo, s)
sum_mpo = mpo + mpo
@test to_matrix(sum_mpo) == to_matrix(mpo) * 2

a = rand(2, 2)
mps = MPS(a)
mps_product = mpo * mps
a_product = contract(mps_product)
@test a .* 2 ≈ a_product

# =============================================================

mps = MPS(Float64.(reshape(1:4, (2, 2))))
operator_mpo = OperatorMPO(2)
add_operator_matrix!(operator_mpo, [1//2 * id 1//2 * id])
add_operator_matrix!(operator_mpo, [id; id])
mpo = MPO(operator_mpo, s)
product = mpo * mps
@test isapprox(product, mps)

mps = MPS(Float64.(reshape(1:4, (2, 2))))
operator_mpo = OperatorMPO(2)
add_operator_matrix!(operator_mpo, [1//3 * id 1//3 * id 1//3 * id])
add_operator_matrix!(operator_mpo, [id; id; id])
mpo = MPO(operator_mpo, s)
product = mpo * mps
@test isapprox(product, mps)
