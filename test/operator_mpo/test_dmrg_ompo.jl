using Test
include("../../src/dmrg/dmrg.jl")
include("../../src/operator_mpo/operator_mpo.jl")
include("../../src/operator_mpo/heisenberg_hamiltonian.jl")

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

matrix_1 = [0.342 0.239; 0.993 0.690]
matrix_2 = [0.360 0.644; 0.863 0.075]

mps_matrix_1 = [-0.32617966 -0.94530779; -0.94530779  0.32617966]
mps_matrix_2 = [-8.48349112e-01 -3.40388699e-04; -9.61127094e-01 4.53939396e-04]

@test isapprox(matrix_1 * matrix_2', mps_matrix_1 * mps_matrix_2',
               atol=0.0000001)

# ======================================================================

h_2 = HeisenbergHamiltonian(2)
mpo = h_2.tensor_mpo

mps = MPS(rand(2, 2))
update!(mps[1], reshape(mps_matrix_1, (size(mps_matrix_1)[1], 1, size(mps_matrix_1)[2])))
update!(mps[2], reshape(mps_matrix_2, (size(mps_matrix_2)..., 1)))

energy = dot(mps, mpo * mps)
@test isapprox(energy, 0.29141788, atol=0.0000001)

dmrg = DMRG(mpo, mps)

initialize_for_left_to_right_sweep!(dmrg)
mps_copy = copy(mps)
@test mps == mps_copy

sweep_left_to_right!(dmrg)
@test !isidentical(mps, mps_copy)

energy = dot(mps, mpo * mps)
@test isapprox(energy, -0.75, atol=0.0000001)

# ======================================================================

mps = MPS(rand(2, 2))
mps[1].tensor = reshape(mps_matrix_1, (size(mps_matrix_1)[1], 1, size(mps_matrix_1)[2]))
mps[2].tensor = reshape(mps_matrix_2, (size(mps_matrix_2)..., 1))
energy = dot(mps, mpo * mps)
@test isapprox(energy, 0.29141788, atol=0.0000001)

dmrg = DMRG(mpo, mps)
initialize_for_left_to_right_sweep!(dmrg)
sweep_left_to_right!(dmrg)
energy = dot(mps, mpo * mps)
@test isapprox(energy, -0.75, atol=0.0000001)

# ======================================================================

mps = MPS(randn(2, 2) .* 10)
energy = dot(mps, mpo * mps)

dmrg = DMRG(mpo, mps)
energy = compute_ground_state!(dmrg)
@test isapprox(energy, -0.75, atol=0.0000001)

# ======================================================================

h_2 = HeisenbergHamiltonian(2)
mpo = h_2.tensor_mpo
a = rand(2, 2)
mps = MPS(a)

dmrg = DMRG(mpo, mps)
energy = compute_ground_state!(dmrg)
@test isapprox(energy, -0.75, atol=0.0000001)

# ======================================================================

h_3 = HeisenbergHamiltonian(3)
mpo = h_3.tensor_mpo
a = rand(2, 2, 2)
mps = MPS(a)

dmrg = DMRG(mpo, mps)
energy = compute_ground_state!(dmrg)
@test isapprox(energy, -1, atol=0.0000001)

# ======================================================================

h_4 = HeisenbergHamiltonian(4)
mpo = h_4.tensor_mpo
a = randn(2, 2, 2, 2) .* 1000
mps = MPS(a)

dmrg = DMRG(mpo, mps)

energy = compute_ground_state!(dmrg)
@test isapprox(energy, -1.616025403, atol=0.0000001)

# ======================================================================

mpo = HeisenbergHamiltonian(5).tensor_mpo
a = randn(2, 2, 2, 2, 2)
mps = MPS(a)

dmrg = DMRG(mpo, mps)

energy = compute_ground_state!(dmrg)
@test isapprox(energy, -1.9278862, atol=0.0000001)

# ======================================================================

mpo = HeisenbergHamiltonian(10).tensor_mpo
a = randn(2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
mps = MPS(a)

dmrg = DMRG(mpo, mps)

energy = compute_ground_state!(dmrg)
@test isapprox(energy, -4.25803520, atol=0.0000001)

# ======================================================================

h_4 = HeisenbergHamiltonian(4)
mpo = h_4.tensor_mpo
a = randn(2, 2, 2, 2) .* 1000
mps = MPS(a, max_bond_dimension=2)

dmrg = DMRG(mpo, mps)

energy = compute_ground_state!(dmrg)
@test isapprox(energy, -1.540488727, atol=0.0000001)

# ======================================================================

mpo = HeisenbergHamiltonian(10).tensor_mpo
a = randn(2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
mps = MPS(a, max_bond_dimension=14)

dmrg = DMRG(mpo, mps)

energy = compute_ground_state!(dmrg)
@test isapprox(energy, -4.25803520, atol=0.00001)

# ======================================================================

mpo = HeisenbergHamiltonian(20).tensor_mpo
a = randn(Tuple([2 for i in 1:mpo.site_count]))
mps = MPS(a, max_bond_dimension=20)
dmrg = DMRG(mpo, mps)
energy = compute_ground_state!(dmrg)
@test isapprox(energy, -8.68247290, atol=0.0000001)

