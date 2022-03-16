using LinearAlgebra
include("../operator_mpo.jl")
include("../molecular_system.jl")

mutable struct OneElectronOperatorMPO

    site_count::Int
    molecular_system::MolecularSystem
    mpos::Array{OperatorMPO, 1}

    function OneElectronOperatorMPO(site_count::Int)

        oe_mpo = new(site_count, MolecularSystem())
        oe_mpo.mpos = Array{OperatorMPO, 1}()

        add_diagonal_mpo!(oe_mpo)

        for i in 1:site_count
            add_off_diagonal_mpo!(oe_mpo, i, :up, :left)
            add_off_diagonal_mpo!(oe_mpo, i, :up, :right)
            add_off_diagonal_mpo!(oe_mpo, i, :down, :left)
            add_off_diagonal_mpo!(oe_mpo, i, :down, :right)
        end

        return oe_mpo

    end
end

function Base.show(io::IO, oe_mpo::OneElectronOperatorMPO)
    site_count = oe_mpo.site_count
    println(io, "$site_count-site OneElectronOperatorMPO with $(length(oe_mpo.mpos)) MPOs")
end

# snajberk 4.49
function add_diagonal_mpo!(oe_mpo::OneElectronOperatorMPO)

    id = oe_mpo.molecular_system.id
    number = oe_mpo.molecular_system.number
    null = oe_mpo.molecular_system.zero
    site_count = oe_mpo.site_count

    t = Array{Sym}(undef, site_count, site_count)

    for c in CartesianIndices(t)
        t[c.I...] = Sym("t_$(join(c.I))")
    end

    mpo = OperatorMPO(site_count)
    add_operator_matrix!(mpo, [id t[1, 1] * number])

    for i in 2:site_count-1
        add_operator_matrix!(mpo, [id t[i, i] * number; null id])
    end

    add_operator_matrix!(mpo, [t[site_count, site_count] * number; id])

    push!(oe_mpo.mpos, mpo)

end

# snajberk 4.75
function add_off_diagonal_mpo!(oe_mpo::OneElectronOperatorMPO, contraction_index::Int,
                               spin::Symbol, side::Symbol)

    id = oe_mpo.molecular_system.id
    number = oe_mpo.molecular_system.number
    null = oe_mpo.molecular_system.zero
    zc = oe_mpo.molecular_system.zc_up_left
    za = oe_mpo.molecular_system.za_up_left

    site_count = oe_mpo.site_count
    s = oe_mpo.molecular_system

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
            add_operator_matrix!(mpo, [id q[i, contraction_index] * zc null;
                                       null s.f r[contraction_index, i] * za;
                                       null null id])
        end

        add_operator_matrix!(mpo, [null; r[contraction_index, site_count] * za; id])

    else

        add_operator_matrix!(mpo, [id r[contraction_index, 1] * za null])

        for i in 2:site_count-1
            add_operator_matrix!(mpo, [id r[contraction_index, i] * za null;
                                       null s.f q[i, contraction_index] * zc;
                                       null null id])
        end

        add_operator_matrix!(mpo, [null; q[site_count, contraction_index] * zc; id])

    end

    push!(oe_mpo.mpos, mpo)

end

function to_symbols(tensor::Array{Float64}, name::String)
    symbol_dict = Dict{Sym, Float64}()
    to_symbols!(symbol_dict, tensor, name)
    return symbol_dict
end

function to_symbols!(symbol_dict::Dict{Sym, Float64}, tensor::Array{Float64}, name::String)
    for c in CartesianIndices(tensor)
        symbol_dict[Sym("$(name)_$(join(c.I))")] = tensor[c.I...]
    end
end

function to_matrix(oe_mpo::OneElectronOperatorMPO, t::Array{Float64, 2})

    s = oe_mpo.molecular_system
    site_count = oe_mpo.site_count
    matrix = zeros(4^site_count, 4^site_count)

    symbol_dict = to_symbols(t, "t")
    q, r = qr(t)
    to_symbols!(symbol_dict, collect(q), "q")
    to_symbols!(symbol_dict, collect(r), "r")

    for mpo in oe_mpo.mpos

        mpo_matrix = to_matrix(mpo, s)
        mpo_matrix = mpo_matrix.xreplace(symbol_dict)

        matrix += N(mpo_matrix.tolist())

    end

    return matrix

end

function to_mpos(oe_mpo::OneElectronOperatorMPO, t::Array{Float64, 2})

    s = oe_mpo.molecular_system
    site_count = oe_mpo.site_count
    mpos = Array{MPO, 1}()

    symbol_dict = to_symbols(t, "t")
    q, r = qr(t)
    to_symbols!(symbol_dict, collect(q), "q")
    to_symbols!(symbol_dict, collect(r), "r")

    for mpo in oe_mpo.mpos
        push!(mpos, MPO(mpo, s, symbol_dict))
    end

    return mpos

end

Base.getindex(oe_mpo::OneElectronOperatorMPO, mpo_index::Int) = oe_mpo.mpos[mpo_index]
Base.lastindex(oe_mpo::OneElectronOperatorMPO) = length(oe_mpo.mpos)
