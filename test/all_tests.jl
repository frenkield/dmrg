include("dmrg/test_dmrg.jl")
include("dmrg/test_left_energy_tensor.jl")
include("dmrg/test_mps.jl")
include("dmrg/test_mps_normalization.jl")
include("dmrg/test_mps_tensor.jl")
include("dmrg/test_right_energy_tensor.jl")
include("dmrg/test_fcidump.jl")
include("dmrg/test_mpo.jl")

# note: as indicated in the readme, the OperatorMPO code (and thus all the
# tests below) can be ignored.

include("operator_mpo/test_operator_mpo.jl")
include("operator_mpo/molecular_mpo/test_one_electron_mpo.jl")
include("operator_mpo/test_heisenberg.jl")
include("operator_mpo/test_molecular_hamiltonian.jl")
include("operator_mpo/test_dmrg_ompo.jl")
