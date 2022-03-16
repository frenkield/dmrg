using LinearAlgebra
using Printf

Base.show(io::IO, f::Float64) = @printf(io, "%1.5f", f)

up = [1
      0]

down = [0
        1]

spin_basis = [up, down]

state_basis = []
state_matrix_basis = []

for i in 1:2
    for j in 1:2

        state = kron(spin_basis[i], spin_basis[j])
        push!(state_basis, state)

        state_matrix = zeros(Int, 2, 2)
        state_matrix[i, j] = 1
        push!(state_matrix_basis, state_matrix)

    end
end


@assert reshape(state_basis[2], (2,2))' == state_matrix_basis[2]


#=

entanglement_entropy2(density_matrix) = sum(density_matrix * log(density_matrix))


function entanglement_entropy(state)

    state_matrix = reshape(state, (2, 2))
    density_matrix = state_matrix * state_matrix'

    singular_values = svd(density_matrix).S
    println(singular_values)

    singular_values = filter(value -> value != 0, singular_values)
    println(singular_values)

    singular_values_squared = singular_values .^ 2
    log_singular_values_squared = log.(singular_values_squared)
    return -1 * dot(singular_values_squared, log_singular_values_squared)
end


state1 = kron(up, up) + kron(down, down)
state1 = state1 / norm(state1)


entropy1 = entanglement_entropy(state1)

=#