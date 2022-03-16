using Test
import ITensors
include("../../src/dmrg/fcidump.jl")
include("../../src/dmrg/itensor_utils.jl")
include("../../src/dmrg/dmrg.jl")

h,v = read_electron_integral_tensors("data/h2/h2.ezfio.FCIDUMP")

sites = ITensors.siteinds("Electron", 2)
one_electron_itensor_mpo = create_one_electron_mpo(sites, h)
two_electron_itensor_mpo = create_two_electron_mpo(sites, v)

one_electron_mpo = MPO(one_electron_itensor_mpo)
two_electron_mpo = MPO(two_electron_itensor_mpo)
molecular_mpo = one_electron_mpo + two_electron_mpo

mps = MPS(randn(4, 4))

energy = compute_ground_state!(molecular_mpo, mps)

nuclear_repulsion_energy = 0.71510433908108118
total_energy = energy + nuclear_repulsion_energy

@test isapprox(total_energy, -1.1372838344894254, atol=0.00000001)

# ==========================================================================

h,v = read_electron_integral_tensors("data/h2o/h2o.ezfio.FCIDUMP")

sites = ITensors.siteinds("Electron", 7)
one_electron_itensor_mpo = create_one_electron_mpo(sites, h)
two_electron_itensor_mpo = create_two_electron_mpo(sites, v)

one_electron_mpo = MPO(one_electron_itensor_mpo)
two_electron_mpo = MPO(two_electron_itensor_mpo)
molecular_mpo = one_electron_mpo + two_electron_mpo

mps = MPS(randn([4 for i in 1:7]...))

energy = compute_ground_state!(molecular_mpo, mps)

nuclear_repulsion_energy = 9.168193300755693
total_energy = energy + nuclear_repulsion_energy

@test isapprox(total_energy, -75.01053577546307, atol=0.01)

# ==========================================================================

mpo = MPO(create_heisenberg_mpo(2))
mps = MPS(randn(2, 2))
dmrg = DMRG(mpo, mps)

energy = compute_ground_state!(dmrg)
@test isapprox(energy, -0.75, atol=0.0000001)

# ==========================================================================

mpo = MPO(create_heisenberg_mpo(3))
mps = MPS(randn(2, 2, 2))

energy = compute_ground_state!(mpo, mps)
@test isapprox(energy, -1.0, atol=0.0000001)

# ==========================================================================

site_count = 10
mpo = MPO(create_heisenberg_mpo(site_count))
mps = MPS(randn([2 for i in 1:site_count]...))

energy = compute_ground_state!(mpo, mps)
@test isapprox(energy, -4.25803520, atol=0.0000001)
