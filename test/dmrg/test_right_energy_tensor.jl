using Test
include("../../src/dmrg/right_energy_tensor.jl")
include("../../src/operator_mpo/operator_mpo.jl")
include("../../src/operator_mpo/spin_system.jl")

s = SpinSystem()

h_2 = OperatorMPO([s.spin_z 1//2 * s.spin_plus 1//2 * s.spin_minus], [s.spin_z; s.spin_minus; s.spin_plus])

mpo = MPO(h_2, s)
mps = MPS(rand(2, 2))
rt = RightEnergyTensor(mpo, mps, 1)

# ===============================================================================

h_3 = OperatorMPO([s.zero  1//2 * s.spin_minus  1//2 * s.spin_plus  s.spin_z  s.id],
                  [s.id s.zero s.zero s.zero s.zero;
                   s.spin_plus s.zero s.zero s.zero s.zero;
                   s.spin_minus s.zero s.zero s.zero s.zero;
                   s.spin_z s.zero s.zero s.zero s.zero
                   s.zero 1//2*s.spin_minus 1//2*s.spin_plus s.spin_z s.id],
                  [s.id; s.spin_plus; s.spin_minus; s.spin_z; s.zero])

mpo = MPO(h_3, s)
mps = MPS(rand(2, 2, 2))
rt = RightEnergyTensor(mpo, mps, 2)
