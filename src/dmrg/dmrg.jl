using KrylovKit
include("left_energy_tensor.jl")
include("right_energy_tensor.jl")

"""
    DMRG

This struct contains all the information required to perform the DRMG
algorithm: the MPO, MPS, left energy tensors (L-expressions), and the
right energy tensors (R-expressions).

After execution of DMRG (via compute_ground_state!()), the MPS contained
in the struct is the ground state of the hamiltonian represented by the
MPO.

Note that the DMRG implementation provided here exactly follows the algorithm
presented in Schollwöck's paper, The density-matrix renormalization group in
the age of matrix product states (https://arxiv.org/abs/1008.3477).
"""
mutable struct DMRG

    mpo::MPO
    mps::MPS
    site_count::Int
    left_energy_tensors::Array{LeftEnergyTensor}
    right_energy_tensors::Array{RightEnergyTensor}

    function DMRG(mpo::MPO, mps::MPS)
        dmrg = new(mpo, mps)
        dmrg.site_count = mpo.site_count
        dmrg.left_energy_tensors = [LeftEnergyTensor()]
        dmrg.right_energy_tensors = [RightEnergyTensor()]
        return dmrg
    end

end

"""
    compute_ground_state!(dmrg::DMRG)

This function encompasses steps 2 through 4 of the DMRG algorithm described on
page 67 of Schollwöck. The initial guess from step 1 is provided by
the caller within the DMRG struct.

The complete DMRG algorithm proceeds as follows:

1. Generate random MPS (the initial guess)
2. Compute right energy tensors (R-expressions)
3. Perform left-to-right sweep (right sweep)
4. Perform right-to-left sweep (left sweep)
5. Terminate when the energy has converged

Note that Schollwöck recommends using the variance of the energy to test
for convergence. Here we just use the energy.

After successful execution the MPS represents the ground state of the
hamiltonian associated with the energy returned by this function.
"""
function compute_ground_state!(dmrg::DMRG)

    initialize_for_left_to_right_sweep!(dmrg)

    energy = 1e30
    previous_energy = 0

    while abs(energy - previous_energy) >= 0.0000001

        previous_energy = energy

        sweep_left_to_right!(dmrg)
        energy = sweep_right_to_left!(dmrg)

    end

    return energy

end

"""
    compute_ground_state!(mpo::MPO, mps::MPS)

This is a convenience method for calling compute_ground_state!(dmrg::DMRG).
"""
function compute_ground_state!(mpo::MPO, mps::MPS)
    dmrg = DMRG(mpo, mps)
    return compute_ground_state!(dmrg)
end

"""
    initialize_right_energy_tensors!(dmrg::DMRG)

Generate the full set of right energy tensors (R-expressions). Right energy tensors
are computed according to expression 193 of Schollwöck.

Note that here we're not computing the right energy tensors recursively as
described in Schollwöck. This isn't a problem here because this is only executed
one at the beginning of the algorithm.

However, in the sweep functions (sweep_left_to_right() and sweep_right_to_left()),
this may be causing performance issues.
"""
function initialize_right_energy_tensors!(dmrg::DMRG)
    @assert length(dmrg.right_energy_tensors) == 1
    for i in 1:dmrg.mps.site_count - 1
        pushfirst!(dmrg.right_energy_tensors, RightEnergyTensor(dmrg.mpo, dmrg.mps, i))
    end
end

"""
    initialize_left_energy_tensors!(dmrg::DMRG)

Generate the full set of left energy tensors (L-expressions). Left energy tensors
are computed according to expression 192 of Schollwöck.

Note that here we're not computing the right energy tensors recursively as
described in Schollwöck. This isn't a problem here because this is only executed
one at the beginning of the algorithm.

However, in the sweep functions (sweep_left_to_right() and sweep_right_to_left()),
this may be causing performance issues.
"""
function initialize_left_energy_tensors!(dmrg::DMRG)
    @assert length(dmrg.left_energy_tensors) == 1
    for i in 1:dmrg.mps.site_count - 1
        push!(dmrg.left_energy_tensors, LeftEnergyTensor(dmrg.mpo, dmrg.mps, i))
    end
end

"""
    initialize_for_right_to_left_sweep!(dmrg::DMRG)

Prepare for execution of the DMRG algorithm by right-normalizing the MPS and
by initializing the full set of right energy tensors.

Note that right-normalization of the MPS involves a succession of reshapes and
SVD decompositions. After right-normalization, the second through the last
tensors of the MPS are those derived (reshaped) from the V' matrices that result
from the successive SVD decompositions. The first tensor of the MPS is derived
from the US (U times singular value matrix) that results from the very last
SVD decomposition.

In other words, right-normalization puts the MPS in tensor train format,
where the process is performed from right to left.
"""
function initialize_for_left_to_right_sweep!(dmrg::DMRG)
    right_normalize!(dmrg.mps)
    initialize_right_energy_tensors!(dmrg)
    dmrg.left_energy_tensors = dmrg.left_energy_tensors[1:1]
end

"""
    compute_partial_hamiltonian(dmrg::DMRG, left_energy_tensor::LeftEnergyTensor,
                                right_energy_tensor::RightEnergyTensor, site_index::Int)

This function computes the (partial) hamiltonian tensor described in Schollwöck on page
66 in the paragraph between expressions 209 and 210.

The hamiltonian tensor is referred to as "partial" because it's the result of a
contraction between the MPO and the MPS where one of the MPS tensors has been
excluded.

In sweep_right_to_left!() and sweep_left_to_right!() the lowest eigenvalue (energy)
and the associated eigenstate are computed for this hamiltonian. The eigenstate
is then reshaped to a tensor and swapped into the MPS.
"""
function compute_partial_hamiltonian(dmrg::DMRG, left_energy_tensor::LeftEnergyTensor,
                                     right_energy_tensor::RightEnergyTensor, site_index::Int)

    left_tensor = left_energy_tensor.tensor
    right_tensor = right_energy_tensor.tensor
    mpo_tensor = dmrg.mpo[site_index]

    @tensor begin
        hamiltonian_tensor[sl, al1, al, slp, al1p, alp] :=
            left_tensor[al1, al1p, bl1] * mpo_tensor[sl, slp, bl1, bl] * right_tensor[al, alp, bl]
    end

    # TODO - Is it appropriate to zero out numerically tiny values in the hamiltonian?
    #        Because of numerical error this hamiltonian isn't hermitian. If small values
    #        are replaced with zeros, the hamiltonian becomes hermitian. And it's possible
    #        that ensuring hermiticity can speed up the eigenvalue search.

    hamiltonian_tensor_size = size(hamiltonian_tensor)
    hamiltonian_size = (prod(hamiltonian_tensor_size[1:3]), prod(hamiltonian_tensor_size[4:6]))
    hamiltonian = reshape(hamiltonian_tensor, hamiltonian_size)

    return hamiltonian

end

"""
    sweep_right_to_left!(dmrg::DMRG)

Perform right-to-left sweep (left sweep). The sweep proceeds as follows:

1. For each site, compute the hamiltonian tensor described in Schollwöck on page
   66 in the paragraph between expressions 209 and 210.

2. Reshape (unfold) the hamiltonian tensor to a matrix

3. Compute the lowest eigenvalue of the hamiltonian matrix and the associated
   eigenvector.

4. Reshape the eigenvector into a tensor.

5. Replace the current tensor (site) of the MPS with the reshaped eigenvector.

6. Normalize the updated MPS. This updates the newly replaced tensor in the MPS,
   and the tensor immediately to its right.

7. Update the right energy tensors (R-expressions).
"""
function sweep_right_to_left!(dmrg::DMRG)

    @assert length(dmrg.left_energy_tensors) == dmrg.site_count "not left initialized"
    @assert length(dmrg.right_energy_tensors) == 1 "not left initialized"

    energy = 0

    for i in 1:dmrg.site_count - 1

        target_site_index = dmrg.site_count - i + 1
        left_energy_tensor = pop!(dmrg.left_energy_tensors)
        right_energy_tensor = dmrg.right_energy_tensors[1]

        hamiltonian =
            compute_partial_hamiltonian(dmrg, left_energy_tensor, right_energy_tensor,
                                        target_site_index)

        start_state = reshape(dmrg.mps[target_site_index].tensor, (:))

        eigenvalues, eigenvectors, info =
            eigsolve(hamiltonian, start_state, 1, :SR, issymmetric=true)

        new_tensor = reshape(eigenvectors[1], size(dmrg.mps[target_site_index]))
        update!(dmrg.mps[target_site_index], real(new_tensor))

        right_normalize_tensor!(dmrg.mps, target_site_index)

        # TODO - Compute the right energy tensor using the previous right energy tensor.
        #        Fixing this may improve performance. However, it's not clear how much
        #        this actually affects performance.
        pushfirst!(dmrg.right_energy_tensors, RightEnergyTensor(dmrg.mpo, dmrg.mps, i))

        energy = eigenvalues[1]

    end

    return energy

end

"""
    sweep_left_to_right!(dmrg::DMRG)

Perform left-to-right sweep (right sweep). The sweep proceeds as follows:

1. For each site, compute the hamiltonian tensor described in Schollwöck on page
   66 in the paragraph between expressions 209 and 210.

2. Reshape (unfold) the hamiltonian tensor to a matrix

3. Compute the lowest eigenvalue of the hamiltonian matrix and the associated
   eigenvector.

4. Reshape the eigenvector into a tensor.

5. Replace the current tensor (site) of the MPS with the reshaped eigenvector.

6. Normalize the updated MPS. This updates the newly replaced tensor in the MPS,
   and the tensor immediately to its left.

7. Update the left energy tensors (L-expressions).
"""
function sweep_left_to_right!(dmrg::DMRG)

    @assert length(dmrg.right_energy_tensors) == dmrg.site_count "not right initialized"
    @assert length(dmrg.left_energy_tensors) == 1 "not right initialized"

    energy = 0

    for i in 1:dmrg.site_count - 1

        target_site_index = i

        right_energy_tensor = popfirst!(dmrg.right_energy_tensors)
        left_energy_tensor = dmrg.left_energy_tensors[end]

        hamiltonian =
            compute_partial_hamiltonian(dmrg, left_energy_tensor, right_energy_tensor,
                                        target_site_index)

        start_state = reshape(dmrg.mps[target_site_index].tensor, (:))

        eigenvalues, eigenvectors, info =
            eigsolve(hamiltonian, start_state, 1, :SR, issymmetric=true)

        new_tensor = reshape(eigenvectors[1], size(dmrg.mps[target_site_index]))
        update!(dmrg.mps[target_site_index], real(new_tensor))

        left_normalize_tensor!(dmrg.mps, target_site_index)

        # TODO - Compute the left energy tensor using the previous left energy tensor.
        #        Fixing this may improve performance. However, it's not clear how much
        #        this actually affects performance.
        push!(dmrg.left_energy_tensors, LeftEnergyTensor(dmrg.mpo, dmrg.mps, i))

        energy = eigenvalues[1]

    end

end
