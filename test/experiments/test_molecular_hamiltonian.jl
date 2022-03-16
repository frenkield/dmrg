using Test
include("../../experiments/molecular_hamiltonian.jl")

@test creation_up * vacuum == up
@test creation_up * down == both
@test creation_up * both == zero

@test creation_down * vacuum == down
@test creation_down * up == both
@test creation_down * both == zero

@test creation_up * annihilation_up * up == up
@test creation_up * annihilation_up * both == both

@test creation_down * annihilation_down * down == down
@test creation_down * annihilation_down * both == both

let
    both_both = creation_down_1 * creation_down_2 * creation_up_1 * creation_up_2 * vacuum_2
    @test both_both == -kron(both, both)

    both_both = creation_down_2 * creation_down_1 * creation_up_1 * creation_up_2 * vacuum_2
    @test both_both == -kron(both, both)
end

# ==============================================================

@test annihilation_up_1 == creation_up_1'
@test annihilation_up_2 == creation_up_2'
@test annihilation_down_1 == creation_down_1'
@test annihilation_down_2 == creation_down_2'

# ==============================================================

@test anti_commutator(creation_up_1, annihilation_up_1) == I(16)
@test anti_commutator(creation_down_1, annihilation_down_1) == I(16)
@test anti_commutator(creation_up_2, annihilation_up_1) == zeros(Int, 16, 16)

@test create_site_operator(c_up, 1, 1) == c_up
@test create_site_operator(c_up, 2, 1) == kron(c_up, id)
@test create_site_operator(c_up, 2, 2) == kron(id_jw, c_up)
@test create_site_operator(c_up, 3, 1) == kron(c_up, id, id)
@test create_site_operator(a_down, 3, 1) == kron(a_down, id, id)
@test create_site_operator(a_down, 3, 2) == kron(id_jw, a_down, id)
@test create_site_operator(a_down, 3, 3) == kron(id_jw, id_jw, a_down)

# TODO - test tensor symmetry
# V = permutedims(V, [4, 2, 3, 1]) + V
# V = permutedims(V, [1, 3, 2, 4]) + V
# V = permutedims(V, [2, 1, 4, 3]) + V
# V = permutedims(V, [3, 4, 1, 2]) + V

# norm(H_2 - H_2')

let

    operator_cache = SiteOperatorCache(2)
    add_operator(operator_cache, :c_up)
    add_operator(operator_cache, :c_down)
    add_operator(operator_cache, :a_up)
    add_operator(operator_cache, :a_down)

    t = reshape(1:4, (2,2))
    H_1 = calculate_1_electron_hamiltonian(operator_cache, t)

    @test H_1 == [
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 2.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 2.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 4.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 1.5 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 2.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 1.5 0.0 0.0 2.5 0.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.5 0.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.0;
        0.0 0.0 1.5 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 1.5 0.0 0.0 0.0 0.0 0.0 2.5 0.0 0.0 1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.5 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.5 0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 -1.5 0.0 0.0 1.5 0.0 0.0 1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.5 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.5 0.0 0.0 3.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0
    ]

    V = reshape(1:16, (2, 2, 2, 2))
    H_2 = calculate_2_electron_hamiltonian(operator_cache, V)
    
    @test H_2 == [
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 16.0 0.0 0.0 -2.0 0.0 0.0 10.0 10.0 0.0 4.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.5 0.0 0.0 8.5 0.0 0.0 2.5 9.0 0.0 0.5 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 24.5 0.0 0.0 0.0 9.0 0.0 -9.5 -10.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 14.5 0.0 0.0 -2.5 0.0 0.0 8.5 8.0 0.0 2.5 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 24.5 0.0 0.0 0.5 0.0;
        0.0 0.0 0.0 13.0 0.0 0.0 -2.0 0.0 0.0 7.0 7.0 0.0 1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 -16.5 0.0 0.0 0.0 7.0 0.0 9.5 8.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 7.5 0.0 0.0 9.5 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 34.0
    ]

end

# ===========================================================

let

    cache = SiteOperatorCache(2)

    add_operator(cache, :c_up)
    add_operator(cache, :c_down)
    add_operator(cache, :a_up)
    add_operator(cache, :a_down)

end


