using LinearAlgebra
using TensorOperations
include("mps_tensor.jl")

"""
    MPS

This struct represents a Matrix Product Operator (MPS). It contains the set of
tensors whose contraction represents a single tensor.

Note that in this implementation of an MPS, all the tensors are of order 3.
In most of the literature (and in ITensor), the first and last tensors
are of order 2. In this implementation the first and last tensors are of
order 3, where one of the dimensions is 1. For example, the first
tensor is 4x1x4, and the last tensor is 4x4x1.

The `physical_dimension` is the dimension of the underlying one-site (one-orbital)
space. In the case of a molecular system the physical dimension is 4. This represents
the 4-dimensional space for a single orbital: vacuum, up-occupied, down-occupied, and
both-occupied (up and down). For a Heisenberg spin chain the physical dimension is 2.
This represents the 2-dimensional space for a single spin: up and down.

The `bond_dimension` determines the maximum length of the dimensions of a
tensor contained in the MPS. During the DMRG algorithm a tensor is
compressed (via SVD truncation) when the dimensions of the tensor grow
beyond the bond dimension.
"""
mutable struct MPS

    site_count::Int64
    tensors::Array{MPSTensor, 1}
    physical_dimension::Int
    max_bond_dimension::Int

    function MPS(site_count::Int64; max_bond_dimension=0)
        mps = new(site_count)
        mps.tensors = Array{MPSTensor, 1}()
        mps.max_bond_dimension = max_bond_dimension
        return mps
    end

end

"""
    random_mps(physical_dimension::Int, site_count::Int; max_bond_dimension=20)

Generate a random (and left-normalized) MPS. This function simply generates an
array of random tensors whoses sizes are restricted by the `bond_dimension.`

The random tensor is left-normalized after generation. Note that this means that
the first `site_count-1` tensors are reshaped U matrices (from an SVD), and that
the last tensor is the reshaped S*V' matrix from the last SVD.
"""
function random_mps(physical_dimension::Int, site_count::Int; max_bond_dimension=20)

    @assert max_bond_dimension >= physical_dimension

    mps = MPS(site_count, max_bond_dimension=max_bond_dimension)
    mps.physical_dimension = physical_dimension

    left_bond_dimension = 1

    for i in 1 : site_count

        up_scale = physical_dimension^i
        down_scale = physical_dimension^(site_count - i)
        right_bond_dimension = min(up_scale, down_scale, max_bond_dimension)

        dimensions = (physical_dimension, left_bond_dimension, right_bond_dimension)
        is_last = i == site_count

        tensor = MPSTensor(randn(dimensions), is_last)
        add_tensor!(mps, tensor)

        left_bond_dimension = right_bond_dimension

    end

    left_normalize!(mps)
    return mps

end

"""
    left_normalize!(mps::MPS)

Update the MPS so that the first `site_count-1` tensors become reshaped U matrices
(from an SVD), and so that the last tensor becomes a reshaped S*V' matrix from the
last SVD.
"""
function left_normalize!(mps::MPS)
    for i in 1:mps.site_count - 1
        left_normalize_tensor!(mps, i)
    end
end

"""
    left_normalize_tensor!(mps::MPS, site_index::Int)

Update a tensor of the MPS (and the tensor to its right) so that the tensor
becomes a reshaped U matrix (from an SVD), and so that the tensor to its
right becomes a reshaped S*V' matrix (from an SVD).
"""
function left_normalize_tensor!(mps::MPS, site_index::Int)
    @assert site_index < mps.site_count
    left_tensor = mps[site_index]
    right_tensor = mps[site_index + 1]
    left_normalize!(left_tensor, right_tensor, mps.max_bond_dimension)
end

"""
    right_normalize!(mps::MPS)

Update the MPS so that the last (rightmost) `site_count-1` tensors are reshaped
V' matrices (from an SVD), and so that the first tensor is the reshaped U*S matrix
(from the last-performed SVD).
"""
function right_normalize!(mps::MPS)
    for i in mps.site_count : -1 : 2
        right_normalize_tensor!(mps, i)
    end
end

"""
    right_normalize_tensor!(mps::MPS, site_index::Int)

Update a tensor of the MPS (and the tensor to its left) so that the tensor
becomes a reshaped V' matrix (from an SVD), and so that the tensor to its
right becomes a reshaped U*S matrix (from an SVD).
"""
function right_normalize_tensor!(mps::MPS, site_index::Int)
    @assert site_index > 1
    left_tensor = mps[site_index - 1]
    right_tensor = mps[site_index]
    right_normalize!(left_tensor, right_tensor, mps.max_bond_dimension)
end

"""
    MPS(full_tensor::Array{Float64}

Convert an arbitrary tensor to an MPS. This performs a tensor-train
transformation of the tensor. The process involves many singular value
decompositions and reshapes.

If `max_bond_dimension` is specified, the matrices resulting from the SVD
are appropriately truncated before being reshaped to tensors and added to
the MPS.

Note that this function is not used in the DMRG algorithm. It is however
very useful for testing and debugging.
"""
function MPS(full_tensor::Array{Float64}; max_bond_dimension=0)

    dimensions = size(full_tensor)
    mps = MPS(length(dimensions))
    mps.physical_dimension = dimensions[1]

    @assert max_bond_dimension == 0 || max_bond_dimension >= mps.physical_dimension

    mps.tensors = Array{MPSTensor, 1}()
    mps.max_bond_dimension = max_bond_dimension

    matrix = full_tensor
    bond_dimension = 1

    for i in 1:mps.site_count - 1

        matrix_width = mps.physical_dimension^(mps.site_count - i)
        matrix_height = length(matrix) ÷ matrix_width
        matrix = reshape(matrix, (matrix_height, matrix_width))

        u, s, v = svd(matrix)
        vt = v'

        if max_bond_dimension > 0 && length(s) > max_bond_dimension
            u = u[:, 1:max_bond_dimension]
            s = s[1:max_bond_dimension]
            vt = vt[1:max_bond_dimension, :]
        end

        tensor = MPSTensor(u, mps.physical_dimension, bond_dimension)
        add_tensor!(mps, tensor)

        bond_dimension = size(tensor)[3]
        matrix = Diagonal(s) * vt

    end

    tensor = MPSTensor(matrix, mps.physical_dimension, bond_dimension, true)
    add_tensor!(mps, tensor)
    return mps

end

"""
    add_tensor!(mps::MPS, mps_tensor::MPSTensor)

Add an MPSTensor to a  MPS.
"""
function add_tensor!(mps::MPS, mps_tensor::MPSTensor)
    @assert length(mps.tensors) < mps.site_count
    push!(mps.tensors, mps_tensor)
end

"""
    compute_value(mps::MPS, position::Tuple)

Compute a specific value of the full tensor represented by an MPS
"""
function compute_value(mps::MPS, position::Tuple)

    @assert length(position) == mps.site_count
    left_vector = mps[1].tensor[position[1], :, :]

    for i in 2:mps.site_count - 1
        right_tensor = mps[i].tensor
        right_matrix = right_tensor[position[i], :, :]
        left_vector = left_vector * right_matrix
    end

    right_vector = mps[end].tensor[position[end], :]
    return dot(left_vector, right_vector)

end

"""
    contract(mps::MPS)

Completely contract all the tensors on an MPS to extract the single tensor
represented by the MPS. The order of this tensor is `site_count.`

Note that this function only works for small MPSs or order 4 or less. This
function has no practical value in DMRG but it useful for testing.
"""
function contract(mps::MPS)

    tensor = undef

    if mps.site_count == 2
        @tensor begin
            tensor[a, b] := mps[1].tensor[a, 1, i] * mps[2].tensor[b, i, 1]
        end

    elseif mps.site_count == 3
        @tensor begin
            tensor[a, b, c] := mps[1].tensor[a, 1, i] *
                mps[2].tensor[b, i, j] * mps[3].tensor[c, j, 1]
        end

    elseif mps.site_count == 4
        @tensor begin
            tensor[a, b, c, d] := mps[1].tensor[a, 1, i] *
                mps[2].tensor[b, i, j] * mps[3].tensor[c, j, k] *
                mps[4].tensor[d, k, 1]
        end

    else
        error("not implemented for larger number of sites")
    end

    return tensor

end

"""
    bond_dimensions(mps::MPS)

Return an array containing the `site_count-1` bond dimensions. Each bond dimension
represents the rank of the contraction index between adjacent tensors.
"""
function bond_dimensions(mps::MPS)
    [size(tensor)[2] for tensor in mps.tensors[2:end]]
end

"""
    bond_dimensions(mps::MPS)

Return the bond dimension of a specific tensor. This is simply the length of the
second index of the tensor.
"""
function bond_dimension(tensor::Array{Float64})
    size(tensor)[2]
end

"""
    isidentical(mps1::MPS, mps2::MPS)

Evaluate the equality of two MPSs.
"""
function isidentical(mps1::MPS, mps2::MPS)

    if mps1.site_count != mps2.site_count
        return false
    end

    for i in 1:mps1.site_count
        if mps1[i] != mps2[i]
            return false
        end
    end

    return true
end

"""
    Base.:(==)(mps1::MPS, mps2::MPS)

Evaluate the equality of two MPSs. This just calls `isidentical().`
"""
function Base.:(==)(mps1::MPS, mps2::MPS)
    return isidentical(mps1, mps2)
end

"""
    Base.isapprox(mps1::MPS, mps2::MPS)

Return true if the specified MPSs are approximately equal. In this case,
"approximately equal" means that the contractions (full tensors) of
of the two MPSs are approximately equal.

Note that this function requires 2 full MPS contractions. So performance
is in general poor.
"""
function Base.isapprox(mps1::MPS, mps2::MPS)
    return contract(mps1) ≈ contract(mps2)
end

"""
    Base.size(mps::MPS)

Return an array containing the dimensions of each tensor contained in the MPS.
"""
function Base.size(mps::MPS)
    [size(tensor) for tensor in mps.tensors]
end

"""
    matrix_size(mps::MPS)

Return an array containing the dimensions of each matrix associated with each
tensor in the MPS. To simplify the code, the MPS stores (via MPSTensor) the
reshaped/unfolded matrix associated with each tensor.
"""
function matrix_size(mps::MPS)
    [size(tensor.matrix) for tensor in mps.tensors]
end

"""
    LinearAlgebra.dot(mps1::MPS, mps2::MPS)

Compute the dot product of two MPSs.
"""
function LinearAlgebra.dot(mps1::MPS, mps2::MPS)

    @assert mps1.site_count == mps2.site_count

    physical_dimension = size(mps1[1])[1]
    indices = [physical_dimension for i in 1:mps1.site_count]
    value = 0.0

    for c in CartesianIndices(Tuple(indices))
        value += compute_value(mps1, c.I) * compute_value(mps2, c.I)
    end

    return value

end

"""
    LinearAlgebra.norm(mps::MPS)

Compute the norm of an MPS. This is the Frobenius norm of the tensor represented
by the MPS.
"""
function LinearAlgebra.norm(mps::MPS)
    return sqrt(dot(mps, mps))
end

"""
    Base.getindex(mps::MPS, tensor_index::Int)

Return the tensor at the specified site index.
"""
Base.getindex(mps::MPS, tensor_index::Int) = mps.tensors[tensor_index]

"""
    Base.getindex(mps::MPS, indices::Int...)

Return the value of the fully contracted MPS at the specified coordinates.
"""
Base.getindex(mps::MPS, indices::Int...) = compute_value(mps, indices)

"""
    Base.lastindex(mps::MPS)

Return the index of the last tensor in the MPS. This is used to reference the
last tensor in the MPS via `mps[end].`
"""
Base.lastindex(mps::MPS) = mps.site_count

"""
    Base.setindex!(mps::MPS, mps_tensor::MPSTensor, index::Int)

Replace the tensor at the specified index with the specified tensor.
"""
function Base.setindex!(mps::MPS, mps_tensor::MPSTensor, index::Int)
    mps.tensors[index] = mps_tensor
end

"""
    Base.copy(mps::MPS)

Return a deep copy of an MPS.
"""
function Base.copy(mps::MPS)
    mps_copy = MPS(mps.site_count)
    mps_copy.tensors = [copy(tensor) for tensor in mps.tensors]
    return mps_copy
end

"""
    update_matrices!(mps::MPS)

Regenerate all the matrices associated with each tensor in the MPS. For convenience,
the MPS maintains (via MPSTensor) the matrix representation of each tensor.

This can be useful for housekeeping after large updates to the tensors contained
in an MPS.
"""
function update_matrices!(mps::MPS)
    for mps_tensor in mps.tensors
        update!(mps_tensor, mps_tensor.tensor)
    end
end

"""
    kronv(vectors::Array{Int64, 1}...)

Compute the Kronecker product of a set of vectors. This returns a tensor whose
order is equal to the length of the array.

Note that this function isn't required for DMRG. However, it's useful for testing.
"""
function kronv(vectors::Array{Int64, 1}...)

    dimensions = Tuple([length(v) for v in vectors])
    tensor = zeros(dimensions)

    for c in CartesianIndices(dimensions)
        tensor[c.I...] = prod([vectors[i][c.I[i]] for i in 1:length(vectors)])
    end

    return tensor

end

"""
    kronv(vectors::Array{Float64, 1}...)

Compute the Kronecker product of a set of vectors. This returns a tensor whose
order is equal to the length of the array.

Note that this function isn't required for DMRG. However, it's useful for testing.
"""
function kronv(vectors::Array{Float64, 1}...)

    dimensions = Tuple([length(v) for v in vectors])
    tensor = zeros(dimensions)

    for c in CartesianIndices(dimensions)
        tensor[c.I...] = prod([vectors[i][c.I[i]] for i in 1:length(vectors)])
    end

    return tensor

end
