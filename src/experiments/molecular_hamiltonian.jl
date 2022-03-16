# ==============================================================
# Molecular hamiltonian on spatial orbital basis
# ==============================================================

using LinearAlgebra
include("site_operator_cache.jl")

anti_commutator(a, b) = a*b + b*a
nnz(a) = count(x -> x != 0, a)

# orbital basis (site basis)
# TODO - c'est juste ce titre ?

zero = [0 0 0 0]'
vacuum = [1 0 0 0]'
up = [0 1 0 0]'
down = [0 0 1 0]'
both = [0 0 0 1]'

c_up = creation_up = [0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
c_down = creation_down = [0 0 0 0; 0 0 0 0; 1 0 0 0; 0 1 0 0]

a_up = annihilation_up = creation_up'
a_down = annihilation_down = creation_down'

# ==============================================================

id = I(4)

# jordan-wigner
id_jw = Diagonal([1, -1, 1, -1])

c_up_1 = creation_up_1 = kron(creation_up, id)
c_up_2 = creation_up_2 = kron(id_jw, creation_up)

c_down_1 = creation_down_1 = kron(creation_down, id)
c_down_2 = creation_down_2 = kron(id_jw, creation_down)

vacuum_2 = kron(vacuum, vacuum)

# ==============================================================

a_up_1 = annihilation_up_1 = kron(annihilation_up, id)
a_up_2 = annihilation_up_2 = kron(id_jw, annihilation_up)
a_down_1 = annihilation_down_1 = kron(annihilation_down, id)
a_down_2 = annihilation_down_2 = kron(id_jw, annihilation_down)

# ==============================================================

# c_up = [c_up_1, c_up_2]
# c_down = [c_down_1, c_down_2]
# a_up = [a_up_1, a_up_2]
# a_down = [a_down_1, a_down_2]

number_1 = c_up_1 * a_up_1 + c_down_1 * a_down_1
number_2 = c_up_2 * a_up_2 + c_down_2 * a_down_2

# ========================================================================

function create_site_operator(basis_operator, site_count, site_index)

    @assert(site_index <= site_count)

    site_operator = basis_operator

    for i in 1 : site_index - 1
        site_operator = kron(id_jw, site_operator)
    end

    for i in site_index + 1 : site_count
        site_operator = kron(site_operator, id)
    end

    return site_operator

end

# ========================================================================

function calculate_1_electron_hamiltonian(operator_cache::SiteOperatorCache, t)

    site_count = size(t)[1]
    space_dimension = 4^site_count
    sites = 1:site_count
    H = zeros(Float64, space_dimension, space_dimension)

    op(basis_operator::Symbol, site_index) =
        get_operator(operator_cache, basis_operator, site_index)

    for i in sites
        for j in sites

            H += 1/2 * t[i, j] * (
                op(:c_up, i) * op(:a_up, j) +
                op(:c_down, i) * op(:a_down, j)
            )

        end
    end

    return H

end

function calculate_2_electron_hamiltonian(operator_cache::SiteOperatorCache, V)

    site_count = size(V)[1]
    space_dimension = 4^site_count
    sites = 1:site_count
    H = zeros(Float64, space_dimension, space_dimension)

    op(basis_operator::Symbol, site_index) =
        get_operator(operator_cache, basis_operator, site_index)

    for i in sites
        for j in sites
            for k in sites
                for l in sites

                    H += 1/2 * V[i, j, k, l] * (
                        op(:c_up, i) * op(:c_up, j) * op(:a_up, k) * op(:a_up, l) +
                        op(:c_down, i) * op(:c_up, j) * op(:a_up, k) * op(:a_down, l) +
                        op(:c_up, i) * op(:c_down, j) * op(:a_down, k) * op(:a_up, l) +
                        op(:c_up, i) * op(:c_down, j) * op(:a_down, k) * op(:a_down, l)
                    )

                end
            end
        end
    end

    return H

end

#t = randn(2, 2)
#V = randn(2, 2, 2, 2)

# site_count = 3

# t = reshape(1:site_count^2, (site_count, site_count))
# H_1 = calculate_1_electron_hamiltonian(t)

# V = reshape(1:site_count^4, (site_count, site_count, site_count, site_count))
# H_2 = calculate_2_electron_hamiltonian(V)

# H = H_1 + H_2

# @show nnz(H), length(H), nnz(H) / length(H)
