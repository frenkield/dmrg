using LinearAlgebra

"""
    MPSTensor

This struct represents a tensor contained in an MPS.

To simplify the code, this struct alao contains the matrix representation of the
MPS tensor. There's a memory cost to this, but it eliminates the need to
constantly reshape/unfold/refold tensors.
"""
mutable struct MPSTensor

    physical_dimension::Int
    left_bond_dimension::Int
    matrix::Array{Float64, 2}
    tensor::Array{Float64, 3}
    right_bond_dimension::Int
    is_last::Bool

    function MPSTensor(matrix::Array{Float64, 2}, physical_dimension::Int, left_bond_dimension::Int,
                       is_last=false)

        mps_tensor = new(physical_dimension, left_bond_dimension, matrix)
        mps_tensor.tensor = matrix_to_left_tensor(mps_tensor, matrix)
        mps_tensor.right_bond_dimension = size(mps_tensor.tensor)[3]
        mps_tensor.is_last = is_last
        return mps_tensor
    end

    function MPSTensor(tensor::Array{Float64, 3}, is_last=false)
        mps_tensor = new()
        mps_tensor.is_last = is_last
        mps_tensor.physical_dimension = size(tensor)[1]
        update!(mps_tensor, tensor)
        return mps_tensor
    end

    function MPSTensor()
        return new()
    end

end

"""
    matrix_to_left_tensor(mps_tensor::MPSTensor, matrix::Array{Float64, 2})

Reshape and permute a matrix into a tensor (after left normalization).
"""
function matrix_to_left_tensor(mps_tensor::MPSTensor, matrix::Array{Float64, 2})
    tensor = reshape(matrix, (mps_tensor.left_bond_dimension, mps_tensor.physical_dimension, :))
    return permutedims(tensor, [2, 1, 3])
end

"""
    matrix_to_right_tensor(mps_tensor::MPSTensor, matrix::Array{Float64, 2})

Reshape and permute a matrix into a tensor (after right normalization).
"""
function matrix_to_right_tensor(mps_tensor::MPSTensor, matrix::Array{Float64, 2})
    tensor = reshape(matrix, (:, mps_tensor.physical_dimension, mps_tensor.right_bond_dimension))
    return permutedims(tensor, [2, 1, 3])
end

"""
    left_normalize!(left_tensor::MPSTensor, right_tensor::MPSTensor, max_bond_dimension=0)

Left normalize an MPSTensor. Left normalization comprises the following steps:

    1. Reshape the tensor into a matrix
    2. Reshape the tensor to the right into a matrix
    3. Multiply the 2 matrices
    4. Compute the SVD of the matrix product from step 3
    5. Truncate the SVD decomposition if the rank of the matrix product is too large
    6. Replace the (left) tensor with the reshaped U matrix from the SVD
    7. Replace the right tensor with the reshaped S*V' matrix from the SVD
"""
function left_normalize!(left_tensor::MPSTensor, right_tensor::MPSTensor, max_bond_dimension=0)

    # no need to actually reshape because we can just use the matrix representation of the tensor
    left_matrix = left_tensor.matrix

    height = size(right_tensor.matrix)[1]
    right_matrix = reshape(right_tensor.matrix, (size(left_matrix)[2], :))

    u,s,v = svd(left_matrix * right_matrix)
    vt = v'

    if max_bond_dimension > 0 && length(s) > max_bond_dimension
        u = u[:, 1:max_bond_dimension]
        s = s[1:max_bond_dimension]
        vt = vt[1:max_bond_dimension, :]
    end

    left_matrix = u
    right_matrix = reshape(Diagonal(s) * vt, (height, :))

    update!(left_tensor, matrix_to_left_tensor(left_tensor, left_matrix))
    update!(right_tensor, matrix_to_right_tensor(right_tensor, right_matrix))

end

"""
    right_normalize!(left_tensor::MPSTensor, right_tensor::MPSTensor, max_bond_dimension=0)

Right normalize an MPSTensor. Right normalization comprises the following steps:

    1. Reshape the tensor into a matrix
    2. Reshape the tensor to the left into a matrix
    3. Multiply the 2 matrices
    4. Compute the SVD of the matrix product from step 3
    5. Truncate the SVD decomposition if the rank of the matrix product is too large
    6. Replace the (right) tensor with the reshaped S*V' matrix from the SVD
    7. Replace the right tensor with the reshaped U matrix from the SVD
"""
function right_normalize!(left_tensor::MPSTensor, right_tensor::MPSTensor, max_bond_dimension=0)

    left_matrix = left_tensor.matrix
    right_matrix = reshape(right_tensor.matrix, (size(left_matrix)[2], :))

    u,s,v = svd(left_matrix * right_matrix)
    vt = v'

    if max_bond_dimension > 0 && length(s) > max_bond_dimension
        u = u[:, 1:max_bond_dimension]
        s = s[1:max_bond_dimension]
        vt = vt[1:max_bond_dimension, :]
    end

    left_matrix = u * Diagonal(s)
    right_matrix = reshape(vt, (:, size(right_matrix)[2]))

    update!(left_tensor, matrix_to_left_tensor(left_tensor, left_matrix))
    update!(right_tensor, matrix_to_right_tensor(right_tensor, right_matrix))

end

"""
    Base.size(mps_tensor::MPSTensor)

Return the dimensions of the tensor represented by the MPSTensor.
"""
function Base.size(mps_tensor::MPSTensor)
    return size(mps_tensor.tensor)
end

"""
    LinearAlgebra.norm(mps_tensor::MPSTensor)

Compute the norm of the tensor represented by this MPS tensor.
"""
function LinearAlgebra.norm(mps_tensor::MPSTensor)
    return norm(mps_tensor.tensor)
end

"""
    Base.copy(mps_tensor::MPSTensor)

Create a deep copy of an MPSTensor.
"""
function Base.copy(mps_tensor::MPSTensor)
    mps_tensor_copy = MPSTensor()
    mps_tensor_copy.physical_dimension = mps_tensor.physical_dimension
    mps_tensor_copy.left_bond_dimension = mps_tensor.left_bond_dimension
    mps_tensor_copy.matrix = copy(mps_tensor.matrix)
    mps_tensor_copy.tensor = copy(mps_tensor.tensor)
    mps_tensor_copy.right_bond_dimension = mps_tensor.right_bond_dimension
    mps_tensor_copy.is_last = mps_tensor.is_last
    return mps_tensor_copy
end

"""
    Base.:(==)(mps_tensor_1::MPSTensor, mps_tensor_2::MPSTensor)

Check equality of two MPSTensors.
"""
function Base.:(==)(mps_tensor_1::MPSTensor, mps_tensor_2::MPSTensor)
    return mps_tensor_1.left_bond_dimension == mps_tensor_2.left_bond_dimension &&
           mps_tensor_1.right_bond_dimension == mps_tensor_2.right_bond_dimension &&
           mps_tensor_1.matrix ≈ mps_tensor_2.matrix &&
           mps_tensor_1.tensor ≈ mps_tensor_2.tensor &&
           mps_tensor_1.is_last == mps_tensor_2.is_last
end

"""
    update!(mps_tensor::MPSTensor, tensor::Array{Float64, 3})

Replace the tensor contained in an MPSTensor. This also updates the matrix
contained in the MPSTensor.
"""
function update!(mps_tensor::MPSTensor, tensor::Array{Float64, 3})

    mps_tensor.tensor = tensor

    tensor_size = size(tensor)
    mps_tensor.left_bond_dimension = tensor_size[2]
    mps_tensor.right_bond_dimension = tensor_size[3]

    if mps_tensor.is_last
        mps_tensor.matrix = reshape(mps_tensor.tensor, (tensor_size[1], :))'
    else
        permuted_tensor = permutedims(mps_tensor.tensor, [2, 1, 3])
        mps_tensor.matrix = reshape(permuted_tensor, (:, tensor_size[3]))
    end

end