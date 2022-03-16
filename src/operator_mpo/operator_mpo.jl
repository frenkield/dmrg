using SymPy
include("../dmrg/mpo.jl")

mutable struct OperatorMPO

    site_count::Int64
    matrices::Array{Array{Sym}, 1}

    function OperatorMPO(site_count::Int64)
        mpo = new(site_count)
        mpo.matrices = Array{Array{Sym}, 1}()
        return mpo
    end

    function OperatorMPO(matrices...)

        mpo = OperatorMPO(length(matrices))

        for matrix in matrices
            add_operator_matrix!(mpo, matrix)
        end

        return mpo
    end

end

abstract type QuantumSystem end

# TODO - traiter la transformation de jordan-wigner

Base.getindex(mpo::OperatorMPO, matrix_index::Int) = mpo.matrices[matrix_index]

function add_operator_matrix!(mpo::OperatorMPO, operator_matrix::Array{Sym})

    @assert length(mpo.matrices) < mpo.site_count "has space for more operator matrices"

    if isempty(mpo.matrices)
        @assert is_row_vector(operator_matrix) "first matrix is a row matrix (row vector)"
    elseif length(mpo.matrices) == mpo.site_count - 1
        @assert is_column_vector(operator_matrix) "last matrix is a column matrix (column vector)"
    else
        @assert is_matrix(operator_matrix) "operator is a matrix"
    end

    push!(mpo.matrices, operator_matrix)

end

function is_row_vector(matrix::Array{Sym})
    return length(size(matrix)) == 2 && size(matrix)[1] == 1
end

function is_column_vector(matrix::Array{Sym})
 return length(size(matrix)) == 1 || size(matrix)[2] == 1
end

function is_matrix(matrix::Array{Sym})
 return length(size(matrix)) == 2
end

function Base.prod(mpo::OperatorMPO)
    sympy.expand(prod(mpo.matrices))[0]
end

function prod_normalized(mpo::OperatorMPO, zero_symbol::Sym)
    product = sympy.expand(prod(mpo.matrices))[0]
    return product.xreplace(Dict(zero_symbol => 0))
end

function Base.show(io::IO, m::MIME"text/latex", mpo::OperatorMPO)

    if length(mpo.matrices) == 0
        show(io, m, "Empty MPO");

    else
        product = mpo.matrices[1]

        for matrix in mpo.matrices[2:end]
            product = sympy.KroneckerProduct(product, matrix)
        end

        show(io, m, product);
    end
end

function Base.show(io::IO, mpo::OperatorMPO)

    if length(mpo.matrices) == 0
        println(io, "Empty MPO");

    else
        product = mpo.matrices[1]

        for matrix in mpo.matrices[2:end]
            product = sympy.KroneckerProduct(product, matrix)
        end

        show(io, MIME"text/plain"(), product);
    end
end

function to_matrix(mpo::OperatorMPO, quantum_system::QuantumSystem)

    left_matrix = mpo.matrices[1]

    for right_matrix in mpo.matrices[2:end]

        height = size(left_matrix)[1]
        width = length(size(right_matrix)) == 1 ? 1 : size(right_matrix)[2]

        product_matrix = Array{SymMatrix}(undef, height, width)

        for i in 1:height
            for j in 1:width

                product = undef

                for k = 1:size(left_matrix)[2]

                    left_operator = left_matrix[i, k]
                    right_operator = right_matrix[k, j]

                    if left_operator isa Sym && left_operator == 0
                       continue
                    elseif right_operator isa Sym && right_operator == 0
                       continue
                    end

                    left_operator = left_operator.xreplace(quantum_system.values)
                    right_operator = right_operator.xreplace(quantum_system.values)

                    if product == undef
                        product = sympy.kronecker_product(left_operator, right_operator)
                    else
                        product += sympy.kronecker_product(left_operator, right_operator)
                    end

                end

                product_matrix[i, j] = product

            end
        end

        left_matrix = product_matrix

    end

    @assert(size(left_matrix) == (1, 1))
    return left_matrix[1, 1]

end

function to_float64_matrix(mpo::OperatorMPO, quantum_system::QuantumSystem)
    matrix = to_matrix(mpo, quantum_system)
    Float64.(N(matrix.tolist()))
end

function to_float64_matrix(mpo::OperatorMPO, quantum_system::QuantumSystem, symbol_dict::Dict{Sym, Float64})
    matrix = to_matrix(mpo, quantum_system)
    matrix = matrix.xreplace(symbol_dict)
    Float64.(N(matrix.tolist()))
end

function Base.length(mpo::OperatorMPO)
    return mpo.site_count
end

Base.lastindex(mpo::OperatorMPO) = mpo.site_count

function Base.:+(mpo1::OperatorMPO, mpo2::OperatorMPO)

    @assert mpo1.site_count == mpo2.site_count
    sum_mpo = OperatorMPO(mpo1.site_count)

    add_operator_matrix!(sum_mpo, cat(mpo1[1], mpo2[1], dims=(2)))

    for i in 2:mpo1.site_count - 1
        add_operator_matrix!(sum_mpo, cat(mpo1[i], mpo2[i], dims=(1,2)))
    end

    add_operator_matrix!(sum_mpo, cat(mpo1[end], mpo2[end], dims=(1)))
    return sum_mpo

end

function Base.copy(mpo::OperatorMPO)
    mpo_copy = OperatorMPO(mpo.site_count)
    mpo_copy.matrices = [copy(matrix) for matrix in mpo.matrices]
    return mpo_copy
end

function MPO(operator_mpo::OperatorMPO, quantum_system::QuantumSystem,
             symbol_values::Dict{Sym, Float64} = Dict{Sym, Float64}())

    mpo = MPO(operator_mpo.site_count)
    add_tensors!(mpo, operator_mpo, quantum_system, symbol_values)
    return mpo
end

function add_tensors!(mpo::MPO, operator_mpo::OperatorMPO, quantum_system::QuantumSystem,
                      symbol_values::Dict{Sym, Float64} = Dict{Sym, Float64}())

    for operator_matrix in operator_mpo.matrices
        add_tensor!(mpo, operator_matrix, quantum_system, symbol_values)
    end
end

function add_tensor!(mpo::MPO, operator_matrix::Array{Sym},
                     quantum_system::QuantumSystem,
                     symbol_dict::Dict{Sym, Float64} = Dict{Sym, Float64}()) where T<: Number

    @assert length(mpo.tensors) < mpo.site_count
    is_first = isempty(mpo.tensors)

    subbed_matrix = operator_matrix.xreplace(quantum_system.values)

    if !isempty(symbol_dict)
        for i in 1:length(subbed_matrix)
            subbed_matrix[i] = subbed_matrix[i].xreplace(symbol_dict)
        end
    end

    tensor_dimensions = contract_dimensions(subbed_matrix, quantum_system)

    tensor_order = length(tensor_dimensions)
    tensor = zeros(Float64, tensor_dimensions)

    if tensor_order == 3

        for i in 1:length(subbed_matrix)
            matrix = Float64.(N(subbed_matrix[i].tolist()))
            tensor[:, :, i] = matrix
        end

        tensor = reshape(tensor, (size(tensor)..., 1))

        if is_first
            tensor = permutedims(tensor, [1, 2, 4, 3])
        end

    elseif tensor_order == 4

        for i in 1:size(subbed_matrix)[1]

            for j in 1:size(subbed_matrix)[2]
                matrix = Float64.(N(subbed_matrix[i, j].tolist()))
                tensor[:, :, i, j] = matrix
            end
        end
    end

    push!(mpo.tensors, tensor)
    return tensor

end

function contract_dimensions(subbed_matrix, quantum_system::QuantumSystem)

    tensor_dimensions = ()

    if size(subbed_matrix)[1] == 1 || size(subbed_matrix)[2] == 1
        tensor_dimensions = (quantum_system.space_dimension,
                             quantum_system.space_dimension, length(subbed_matrix))
    else
        tensor_dimensions = (quantum_system.space_dimension, quantum_system.space_dimension,
                             size(subbed_matrix)[1], size(subbed_matrix)[2])
    end

    return tensor_dimensions

end

