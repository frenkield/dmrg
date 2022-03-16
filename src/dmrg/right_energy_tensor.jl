using TensorOperations
include("mps.jl")
include("mpo.jl")

"""
    LeftEnergyTensor

This struct represents the L-expression tensor given in expression 193 of
Schollwöck.

This implementation can be improved by building each RightEnergyTensor
recursively. For example, the RightEnergyTensor for the 4th site/orbital
can be constructed by contracting the 5th RightEnergyTensor with the 4th
MPO tensor and the 4th MPS tensor.

TODO - build RightEnergyTensors recursively.
"""
mutable struct RightEnergyTensor

    tensor::Array{Float64, 3}

    RightEnergyTensor() = new(ones(Float64, 1, 1, 1))

    function RightEnergyTensor(mpo::MPO, mps::MPS, site_count::Int)

        @assert site_count < mpo.site_count
        right_energy_tensor = new()

        mpo1 = mpo[end][:, :, :]
        mps1 = mps[end].tensor[:, :]

        @tensor begin
            contraction[a1, a2, b] := mps1[s, a1] * mpo1[s, t, b] * mps1[t, a2]
        end

        start_index = mpo.site_count - 1
        end_index = mpo.site_count - site_count + 1

        for i in start_index : -1 : end_index

            @tensor begin
                contraction[ai1, ai2, bi] := contraction[a1, a2, b] *
                    mps[i].tensor[s, ai1, a1] * mpo[i][s, t, bi, b] * mps[i].tensor[t, ai2, a2]
            end
        end

        right_energy_tensor.tensor = contraction
        return right_energy_tensor
    end

end

# This function isn't needed because the lowest eigenvalue (energy) computed during DMRG
# sweeps is the correct way to test for convergence. It's left here as an example of how
# one can efficiently compute the energy of a state using the left/right energy tensors.

# function compute_energy(left_energy_tensors::Array{LeftEnergyTensor}, mpo::MPO, mps::MPS)
#
#     @assert length(left_energy_tensors) == mpo.site_count
#     left_energy_tensor = left_energy_tensors[end].tensor
#
#     # this includes the extra mpo and mps needed for full energy calculation
#     extended_right_energy_tensor = RightEnergyTensor(mpo, mps, 1).tensor
#
#     @tensor begin
#         energy[] := left_energy_tensor[a, ap, b] * extended_right_energy_tensor[a, ap, b]
#     end
#
#     return energy[1]
#
# end

