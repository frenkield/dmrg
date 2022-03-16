using TensorOperations
include("mps.jl")
include("mpo.jl")

"""
    LeftEnergyTensor

This struct represents the L-expression tensor given in expression 192 of
Schollwöck.

This implementation can be improved by building each LeftEnergyTensor
recursively. For example, the LeftEnergyTensor for the 3rd site/orbital
can be constructed by contracting the 2nd LeftEnergyTensor with the 3rd
MPO tensor and the 3rd MPS tensor.

TODO - build LeftEnergyTensors recursively.
"""
mutable struct LeftEnergyTensor

    tensor::Array{Float64, 3}
    LeftEnergyTensor() = new(ones(Float64, 1, 1, 1))

    function LeftEnergyTensor(mpo::MPO, mps::MPS, site_count::Int)

        @assert site_count < mpo.site_count
        left_energy_tensor = new()

        mpo1 = mpo[1][:, :, 1, :]
        mps1 = mps[1].tensor[:, 1, :]

        @tensor begin
            contraction[a1, a2, b] := mps1[s, a1] * mpo1[s, t, b] * mps1[t, a2]
        end

        for i in 2:site_count
            @tensor begin
                contraction[ai1, ai2, bi] := contraction[a1, a2, b] *
                    mps[i].tensor[s, a1, ai1] * mpo[i][s, t, b, bi] * mps[i].tensor[t, a2, ai2]
            end
        end

        left_energy_tensor.tensor = contraction
        return left_energy_tensor
    end

end

# This function isn't needed because the lowest eigenvalue (energy) computed during DMRG
# sweeps is the correct way to test for convergence. It's left here as an example of how
# one can efficiently compute the energy of a state using the left/right energy tensors.

# function compute_energy(right_energy_tensors::Array{RightEnergyTensor}, mpo::MPO, mps::MPS)
#
#     @assert length(right_energy_tensors) == mpo.site_count
#     right_energy_tensor = right_energy_tensors[1].tensor
#
#     # this includes the extra mpo and mps needed for full energy calculation
#     extended_left_energy_tensor = LeftEnergyTensor(mpo, mps, 1).tensor
#
#     @tensor begin
#         energy[] := right_energy_tensor[a, ap, b] * extended_left_energy_tensor[a, ap, b]
#     end
#
#     return energy[1]
#
# end

