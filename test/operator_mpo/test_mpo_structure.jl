using Test
using SymPy
include("../../src/dmrg/mpo.jl")

#==
 = These DMRG tests specifically use OperatorMPO to generate MPOs.
 = The OperatorMPO is an implementation of the MPO construction method
 = described in Snajberk.
 =
 = In dmrg/test_dmrg.jl we instead use ITensor to construct MPOs.
 = Needless to say, the ITensor method is vastly superior.
 =
 = Long story short, all the OperatorMPO code (including this test) isn't particularly
 = useful.
 =#

function random_operator_matrix(operators, dimensions...)
    [operators[rand(1:length(operators))] for c in CartesianIndices(dimensions)]
end

function generate_symbols(site_count::Int)

    i = [sympy.Symbol("i_$i", commutative=False) for i in 1:site_count]
    a = [sympy.Symbol("a_$i", commutative=False) for i in 1:site_count]
    b = [sympy.Symbol("b_$i", commutative=False) for i in 1:site_count]

    # t = [Sym("t_$(join(c.I))") for c in CartesianIndices((8, 8))]

    return i, a, b

end

function random_mpo_3(site_count::Int)

    i,a,b = generate_symbols(site_count)

    mpo = OperatorMPO(site_count)
    add_operator_matrix!(mpo, [i[1] a[1] b[1]])

    for s in 2:site_count-1
        add_operator_matrix!(mpo, random_operator_matrix([i[s] a[s] b[s]], 3, 3))
    end

    add_operator_matrix!(mpo, [i[end]; a[end]; b[end]])
    return mpo

end

function random_mpo(site_count::Int, width::Int)

    i,a,b = generate_symbols(site_count)
    ops = [i, a, b]
    mpo = OperatorMPO(site_count)

    first_matrix = reshape([ops[i % length(ops) + 1][1] for i in 0:width-1], (1, width))
    add_operator_matrix!(mpo, first_matrix)

    index_matrix = [rand(1:3) for c in CartesianIndices((width, width))]

    for s in 2:site_count-1
        operator_matrix = [ops[index_matrix[c]][s] for c in CartesianIndices(index_matrix)]
        add_operator_matrix!(mpo, operator_matrix)
    end

    first_matrix = [ops[i % length(ops) + 1][end] for i in 0:width-1]
    add_operator_matrix!(mpo, first_matrix)

    return mpo

end

function random_mpo_consistent_4(site_count::Int)

    i,a,b = generate_symbols(site_count)
    operators = [i, a, b]

    mpo = OperatorMPO(site_count)
    add_operator_matrix!(mpo, [i[1] a[1] b[1] i[1]])
    index_matrix = [rand(1:3) for c in CartesianIndices((site_count, site_count))]

    for s in 2:site_count-1
        operator_matrix = [operators[index_matrix[c]][s] for c in CartesianIndices((4, 4))]
        add_operator_matrix!(mpo, operator_matrix)
    end

    add_operator_matrix!(mpo, [i[end]; a[end]; b[end]; i[end]])
    return mpo

end

function to_string_matrix(matrix::Array{Sym})

    string_matrix = string.(matrix)

    for i in 1:length(string_matrix)
        string_matrix[i] = replace(string_matrix[i], r"^[1-9]\*|_[1-9]\*?" => "")
    end

    return string_matrix

end

function extract_terms(mpo::OperatorMPO)

    terms = [string.(prod(mpo).args)...]

    for i in 1:length(terms)
        terms[i] = replace(terms[i], r"^[1-9]\*|_[1-9]\*?" => "")
    end

    return sort(terms)

end

function validate_mpo(mpo::OperatorMPO)

    terms = extract_terms(mpo)

    terms = filter(term -> occursin(r"a.*b|b.*a", term), terms)
    terms = filter(term -> occursin(r"i.*i", term), terms)

    return sort(terms)

end

function find_candidate_mpos(generate_mpo)

    for i in 1:100000

        if i % 1000 == 0
            println(i)
        end

        mpo = generate_mpo()
        p = prod(mpo)

        if length(p.args) >= 30

            terms = validate_mpo(mpo)

            if length(terms) >= 12

                println("=================")

                display(mpo)

                @show length(p.args)

                println(mpo[1])
                println(mpo[2])
                println(mpo[3])
                println(mpo[4])

            end
        end

    end
end

function generate_candidate_mpo_4_1()

    i,a,b = generate_symbols(4)

    mpo = OperatorMPO(4)

    add_operator_matrix!(mpo, [i[1] a[1] b[1] i[1]])
    add_operator_matrix!(mpo, [a[2] i[2] i[2] i[2]; b[2] a[2] i[2] i[2]; i[2] a[2] b[2] i[2]; b[2] b[2] a[2] b[2]])
    add_operator_matrix!(mpo, [a[3] a[3] i[3] i[3]; i[3] b[3] b[3] i[3]; a[3] i[3] i[3] b[3]; a[3] i[3] a[3] b[3]])
    add_operator_matrix!(mpo, [i[4], a[4], b[4], i[4]])

    for c in [(1, 1), (3, 1), (4, 1), (2, 2), (4, 2), (1, 3), (3, 3), (2, 4)]
        mpo[2][c...] = 0
    end

    for c in [(1, 1), (2, 1), (3, 1), (1, 2), (3, 2), (1, 3), (2, 3), (4, 4)]
        mpo[3][c...] = 0
    end

    return mpo

end

function generate_candidate_mpo_4_2()

    i,a,b = generate_symbols(4)

    mpo = OperatorMPO(4)


    add_operator_matrix!(mpo, [i[1] a[1] b[1] i[1]])
    add_operator_matrix!(mpo, [i[2] a[2] i[2] b[2]; i[2] b[2] i[2] b[2]; b[2] i[2] a[2] i[2]; b[2] i[2] a[2] a[2]])
    add_operator_matrix!(mpo, [i[3] a[3] i[3] b[3]; i[3] b[3] i[3] b[3]; b[3] i[3] a[3] i[3]; b[3] i[3] a[3] a[3]])
    add_operator_matrix!(mpo, [i[4], a[4], b[4], i[4]])


    for c in [(1, 1), (3, 1), (4, 1), (3, 2), (2, 3), (4, 3), (2, 4), (4, 4)]
        mpo[2][c...] = 0
    end

    for c in [(1, 1), (3, 1), (4, 1), (1, 2), (3, 2), (4, 3)]
        mpo[3][c...] = 0
    end

    return mpo

end

function shrink_mpo(mpo::OperatorMPO)

    mpo = copy(mpo)

    for matrix_index in 1:4

        println("matrix $matrix_index")

        for c in CartesianIndices(size(mpo[matrix_index]))

            mpo_copy = copy(mpo)
            mpo[matrix_index][c] = 0
            terms = validate_mpo(mpo)

            if length(terms) == 12
                println(c.I)
            else
                mpo = copy(mpo_copy)
            end
        end
    end

    return mpo
end
