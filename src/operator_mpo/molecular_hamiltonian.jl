using LinearAlgebra
include("operator_mpo.jl")
include("molecular_system.jl")
include("../../src/dmrg/mpo.jl")

# TODO - eliminer tout ca
s = MolecularSystem()
@unpack id, c_up, c_down, a_up, a_down, number = s
null = s.zero
@vars ϵ_1 ϵ_2 ϵ_3

# =============================================================

function to_string(c::CartesianIndex)
    return join(c.I)
end

# snajberk 4.49
function one_electron_diagonal(site_count::Int)

    t = Array{Sym}(undef, site_count, site_count)

    for c in CartesianIndices(t)
        t[c.I...] = Sym("t_$(join(c.I))")
    end

    mpo = OperatorMPO(site_count)

    add_operator_matrix!(mpo, [id t[1, 1] * number])

    for i in 2:site_count-1
        add_operator_matrix!(mpo, [id t[i, i] * number; 0 id])
    end

    add_operator_matrix!(mpo, [t[site_count, site_count] * number; id])

    return mpo

end



# ==========================================
# snajberk 5.13 (3.63)
# c'est bon pour jordan-wigner ?
# ==========================================


# snajberk 4.75
function one_electron_off_diagonal(site_count::Int, contraction_index::Int,
                                   spin::Symbol, side::Symbol)

    zc = s.zc_up_left
    za = s.za_up_left

    if spin == :down && side == :left
        zc = s.zc_down_left
        za = s.za_down_left

    elseif spin == :up && side == :right
        zc = s.zc_up_right
        za = s.za_up_right

    elseif spin == :down && side == :right
        zc = s.zc_down_right
        za = s.za_down_right
    end

    q = Array{Sym}(undef, site_count, site_count)
    r = Array{Sym}(undef, site_count, site_count)

    for c in CartesianIndices(q)
        q[c.I...] = Sym("q_$(join(c.I))")
        r[c.I...] = Sym("r_$(join(c.I))")
    end

    mpo = OperatorMPO(site_count)

    if side == :left

        add_operator_matrix!(mpo, [id q[1, contraction_index] * zc null])

        for i in 2:site_count-1
            add_operator_matrix!(mpo, [id q[i, contraction_index] * zc 0;
                                       0 s.f r[contraction_index, i] * za;
                                       0 0 id])
        end

        add_operator_matrix!(mpo, [null; r[contraction_index, site_count] * za; id])

    else

        add_operator_matrix!(mpo, [id r[contraction_index, 1] * za null])

        for i in 2:site_count-1
            add_operator_matrix!(mpo, [id r[contraction_index, i] * za 0;
                                       0 s.f q[i, contraction_index] * zc;
                                       0 0 id])
        end

        add_operator_matrix!(mpo, [null; q[site_count, contraction_index] * zc; id])

    end

    return mpo

end


function one_electron_mpos(site_count::Int)

    mpos = OperatorMPO[]

    push!(mpos, one_electron_diagonal(site_count))

    for i in 1:site_count
        push!(mpos, one_electron_off_diagonal(site_count, i, :up, :left))
        push!(mpos, one_electron_off_diagonal(site_count, i, :up, :right))
        push!(mpos, one_electron_off_diagonal(site_count, i, :down, :left))
        push!(mpos, one_electron_off_diagonal(site_count, i, :down, :right))
    end

    return mpos

end

function to_symbols(tensor::Array{Float64}, name::String)

    symbols = Dict{String, Float64}()

    for c in CartesianIndices(tensor)
        symbols["$(name)_$(join(c.I))"] = tensor[c.I...]
    end

    return symbols

end

function to_matrix(mpos::Array{OperatorMPO}, t::Array{Float64})

    site_count = size(t)[1]
    matrix = zeros(4^site_count, 4^site_count)

    t_symbols = to_symbols(t, "t")
    q, r = qr(t)
    q_symbols = to_symbols(collect(q), "q")
    r_symbols = to_symbols(collect(r), "r")

    for mpo in mpos

        mpo_matrix = to_matrix(mpo, s)
        mpo_matrix = mpo_matrix.subs(t_symbols)
        mpo_matrix = mpo_matrix.subs(q_symbols)
        mpo_matrix = mpo_matrix.subs(r_symbols)

        matrix += N(mpo_matrix.tolist())

    end

    return matrix

end
