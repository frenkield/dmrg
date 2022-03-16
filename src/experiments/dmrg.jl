using LinearAlgebra

u = [
    1
    0
]

d = [
    0
    1
]

S_z = [
    1/2 0
    0 -1/2
]

S_x = [
    0 1/2
    1/2 0
]

S_y = [
    0 -im/2
    im/2 0
]



S_p = real(S_x + im * S_y)
S_m = real(S_x - im * S_y)

H_2 = kron(S_z, S_z) + 1/2 * (kron(S_p, S_m) + kron(S_m, S_p))
H_2 = real(H_2)

H_3 = kron(S_z, S_z, I(2)) + 1/2 * (kron(S_p, S_m, I(2)) + kron(S_m, S_p, I(2))) +
      kron(I(2), S_z, S_z) + 1/2 * (kron(I(2), S_p, S_m) + kron(I(2), S_m, S_p))

H_3 = real(H_3)






H_4 = kron(S_z, S_z, I(2), I(2)) + 1/2 * (kron(S_p, S_m, I(2), I(2)) + kron(S_m, S_p, I(2), I(2))) +
      kron(I(2), S_z, S_z, I(2)) + 1/2 * (kron(I(2), S_p, S_m, I(2)) + kron(I(2), S_m, S_p, I(2))) +
      kron(I(2), I(2), S_z, S_z) + 1/2 * (kron(I(2), I(2), S_p, S_m) + kron(I(2), I(2), S_m, S_p))

H_4 = real(H_4)



H_6 = kron(S_z, S_z, I(2), I(2), I(2), I(2)) +
      1/2 * (kron(S_p, S_m, I(2), I(2), I(2), I(2)) + kron(S_m, S_p, I(2), I(2), I(2), I(2))) +
      kron(I(2), S_z, S_z, I(2), I(2), I(2)) +
      1/2 * (kron(I(2), S_p, S_m, I(2), I(2), I(2)) + kron(I(2), S_m, S_p, I(2), I(2), I(2))) +
      kron(I(2), I(2), S_z, S_z, I(2), I(2)) +
      1/2 * (kron(I(2), I(2), S_p, S_m, I(2), I(2)) + kron(I(2), I(2), S_m, S_p, I(2), I(2))) +
      kron(I(2), I(2), I(2), S_z, S_z, I(2)) +
      1/2 * (kron(I(2), I(2), I(2), S_p, S_m, I(2)) + kron(I(2), I(2), I(2), S_m, S_p, I(2))) +
      kron(I(2), I(2), I(2), I(2), S_z, S_z) +
      1/2 * (kron(I(2), I(2), I(2), I(2), S_p, S_m) + kron(I(2), I(2), I(2), I(2), S_m, S_p))

H_6 = real(H_6)






# eigen(H_2).values

# ===================================

S_z_total = kron(S_z, S_z)

p = kron(u, u)
m = kron(d, d)

s = kron(u, d) - kron(d, u)
t = kron(u, d) + kron(d, u)

#@show S_z_total * s
#@show S_z_total * t


# ===================================




#=
julia> G = G * (1/sqrt(2))
2×2 Array{Float64,2}:
  0.0       0.707107
 -0.707107  0.0
=#


function entanglement_entropy(singular_values)
  singular_values_squared = singular_values .^ 2
  log_singular_values_squared = log.(singular_values_squared)
 -1 * dot(singular_values_squared, log_singular_values_squared)
end

function basis(dimension, index)
    [i == index for i in 1:dimension]
end

function compute_half_matrix(hamiltonian)

    ground_state = eigen(hamiltonian).vectors[:,1]
    dimension = Int(sqrt(size(hamiltonian)[1]))
    G = zeros(dimension, dimension)

    for i in 1:dimension
      for j in 1:dimension

        basis_vector = kron(basis(dimension, i), basis(dimension, j))
        G[i, j] = dot(basis_vector', ground_state)

      end
    end

    return G

end

half_matrix = compute_half_matrix(H_6)
singular_values = svd(half_matrix).S

density_matrix = half_matrix * half_matrix'


# et puis coupure !!!!!!


entropy = entanglement_entropy(singular_values)


