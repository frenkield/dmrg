using TensorOperations
import ITensors
include("mps.jl")

"""
    MPO

This struct represents a Matrix Product Operator (MPO). It contains a
set of tensors whose contraction is a hamiltonian tensor.

This MPO implementation does not do any compression. However, the easiest
way to create an MPO is to convert an ITensors.MPO to an MPO. So, because
ITensor compresses MPOs, one can assume that this MPO is compressed.

Note that in this implementation of an MPO, all the tensors are of order 4.
In most of the literature (and in ITensor), the first and last tensors
are of order 3. In this implementation the first and last tensors are of
order 4, where one of the dimensions is 1. For example, the first
tensor is 4x4x1x3, and the last tensor is 4x4x3x1.
"""
mutable struct MPO

    site_count::Int64
    tensors::Array{Array{Float64}, 1}

    function MPO(site_count::Int64)
        mpo = new(site_count)
        mpo.tensors = Array{Array{Float64}, 1}()
        return mpo
    end

    function MPO(itensor_mpo::ITensors.MPO)
        mpo = MPO(length(itensor_mpo))
        import_itensor_mpo!(mpo, itensor_mpo)
        return mpo
    end

    function MPO(tensors::Array{Array{Float64}, 1})
        mpo = MPO(length(tensors))
        mpo.tensors = tensors
        return mpo
    end

end

"""
    import_itensor_mpo!(mpo::MPO, itensor_mpo::ITensors.MPO)

Convert an ITensors.MPO to an MPO. This is simply a direct conversion of the
ITensors (in the ITensors.MPO) to normal Julia multidimensional arrays.
"""
function import_itensor_mpo!(mpo::MPO, itensor_mpo::ITensors.MPO)

    @assert length(itensor_mpo) >= 2
    @assert length(itensor_mpo) == length(mpo)

    tensor = itensor_tensor_to_array(itensor_mpo[1])
    tensor = permutedims(reshape(tensor, (size(tensor)..., 1)), [2, 3, 4, 1])

    push!(mpo.tensors, tensor)

    for i in 2:length(mpo)-1

        tensor = itensor_tensor_to_array(itensor_mpo[i])
        tensor = permutedims(tensor, [3, 4, 1, 2])
        push!(mpo.tensors, tensor)

    end

    tensor = itensor_tensor_to_array(itensor_mpo[end])
    tensor = permutedims(reshape(tensor, (size(tensor)..., 1)), [2, 3, 1, 4])
    push!(mpo.tensors, tensor)

end

"""
    compute_tensor_value(mpo::MPO, position::Tuple)

Compute a element of the tensor represented by the MPO.
"""
function compute_tensor_value(mpo::MPO, position::Tuple)

    @assert length(position) == mpo.site_count * 2
    left_vector = mpo.tensors[1][position[1], position[2], 1, :]

    for right_tensor in mpo.tensors[2:end-1]
        right_matrix = right_tensor[position[3], position[4], :, :]
        left_vector = left_vector' * right_matrix
    end

    right_vector = mpo.tensors[end][position[end-1], position[end], :]

    value = sum(left_vector .* right_vector)
    return value
end

"""
    Base.:*(mpo::MPO, mps::MPS)

Compute the product of MPO by an MPS (mpo * mps).
"""
function Base.:*(mpo::MPO, mps::MPS)

    @assert mpo.site_count == mps.site_count
    product = MPS(mps.site_count)

    @tensor begin
        contraction[i, b, a] := mpo[1][i, j, 1, b] * mps[1].tensor[j, 1, a]
    end

    dimensions = size(contraction)
    tensor = reshape(contraction, (dimensions[1], 1, dimensions[2] * dimensions[3]))
    add_tensor!(product, MPSTensor(tensor))

    for s in 2 : mpo.site_count-1

        @tensor begin
            contraction[l, a, b, c, d] := mpo[s][l, k, a, b] * mps[s].tensor[k, c, d]
        end

        contraction = permutedims(contraction, (1, 2, 4, 3, 5))

        dimensions = size(contraction)
        dimensions = (dimensions[1], dimensions[2] * dimensions[3], dimensions[4] * dimensions[5])
        add_tensor!(product, MPSTensor(reshape(contraction, dimensions)))

    end

    @tensor begin
        contraction[i, b, a] := mpo[end][i, j, b, 1] * mps[end].tensor[j, a, 1]
    end

    dimensions = size(contraction)
    tensor = reshape(contraction, (dimensions[1], dimensions[2] * dimensions[3], 1))
    add_tensor!(product, MPSTensor(tensor, true))

    return product

end

"""
    Base.size(mpo::MPO)

Return an array containing the sizes of all the tensors contained in the MPO.
"""
function Base.size(mpo::MPO)
    [size(tensor) for tensor in mpo.tensors]
end

"""
    Base.:*(mpo1::MPO, mpo2::MPO)

Compute the product of two MPOs (mpo1 * mpo2).
"""
function Base.:*(mpo1::MPO, mpo2::MPO)

    @assert mpo1.site_count == mpo2.site_count
    product_mpo = MPO(mpo1.site_count)

    for i in 1 : length(mpo1.tensors)

        tensor1 = mpo1[i]
        size1 = size(tensor1)
        tensor2 = mpo2[i]
        size2 = size(tensor2)
        physical_dimension = size1[1]

        dimensions = (physical_dimension, physical_dimension, size1[3] * size2[3], size1[4] * size2[4])
        product_tensor = zeros(dimensions)

        for i in 1:physical_dimension
            for j in 1:physical_dimension
                for n in 1:physical_dimension
                    product_tensor[i, j, :, :] += kron(tensor1[i, n, :, :], tensor2[n, j, :, :])
                end
            end
        end

        push!(product_mpo.tensors, product_tensor)

    end

    return product_mpo

end

"""
    Base.getindex(mpo::MPO, tensor_index::Int)

Return the tensor at the index `tensor_index.`
"""
Base.getindex(mpo::MPO, tensor_index::Int) = mpo.tensors[tensor_index]

"""
    Base.lastindex(mpo::MPO)

Return the index of the last tensor in the MPO. This provides access to
tensors in the MPO via `mpo[end].`
"""
Base.lastindex(mpo::MPO) = mpo.site_count

"""
    Base.:+(mpo1::MPO, mpo2::MPO)

Compute the sum of two MPOs (mpo1 + mpo2).
"""
function Base.:+(mpo1::MPO, mpo2::MPO)

    @assert mpo1.site_count == mpo2.site_count
    sum_mpo = MPO(mpo1.site_count)

    # treat the first tensor as a 3rd order tensor
    sum_tensor = cat(mpo1[1], mpo2[1], dims=(4))
    push!(sum_mpo.tensors, sum_tensor)

    for i in 2 : length(mpo1.tensors)-1
        sum_tensor = cat(mpo1[i], mpo2[i], dims=(3,4))
        push!(sum_mpo.tensors, sum_tensor)
    end

    # treat the last tensor as a 3rd order tensor
    sum_tensor = cat(mpo1[end], mpo2[end], dims=(3))
    push!(sum_mpo.tensors, sum_tensor)

    return sum_mpo

end

"""
    Base.show(io::IO, mpo::MPO)

Show all the tensors contained in an MPO.
"""
function Base.show(io::IO, mpo::MPO)
    if length(mpo.tensors) == 0
        println(io, "Empty MPO")
    else
        for tensor in mpo.tensors
            display(tensor)
        end
    end
end

"""
    Base.length(mpo::MPO)

Return the number of tensors contained in the MPO (site count).
"""
function Base.length(mpo::MPO)
    return mpo.site_count
end

"""
    contract(mpo::MPO)

Contract MPO into a single tensor. This is only useful for testing. In particular,
this function only works on small MPOs with up to 3 sites.
"""
function contract(mpo::MPO)

    tensor = undef

    if mpo.site_count == 2
        @tensor begin
            tensor[a, b, c, d] := mpo.tensors[1][a, b, 1, i] * mpo.tensors[2][c, d, i, 1]
        end

    elseif mpo.site_count == 3
        @tensor begin
            tensor[a, b, c, d, e, f] := mpo.tensors[1][a, b, 1, i] *
                mpo.tensors[2][c, d, i, j] * mpo.tensors[3][e, f, j, 1]
        end

    else
        error("not implemented for more than 3 sites")

    end

    return tensor

end

"""
    to_matrix(mpo::MPO)

Contract MPO into a matrix. This is only useful for testing. In particular,
this function only works on small MPOs with up to 3 sites.
"""
function to_matrix(mpo::MPO)

    basis_length = size(mpo[1])[1]
    matrix_size = basis_length^mpo.site_count
    tensor = contract(mpo)

    if mpo.site_count == 2
        return reshape(permutedims(tensor, [3, 1, 4, 2]), (matrix_size, matrix_size))
    elseif mpo.site_count == 3
        return reshape(permutedims(tensor, [5, 3, 1, 6, 4, 2]), (matrix_size, matrix_size))
    else
        error("not implemented for more than 3 sites")
    end

end
