using LinearAlgebra
include("one_electron_operator_mpo.jl")
include("../../dmrg/mpo.jl")

# TODO - jordan-wigner, c'est deja la grace aux operateurs droite/gauche ???
mutable struct OneElectronMPO

    mpos::Array{MPO, 1}

    function OneElectronMPO(oe_integrals_tensor::Array{Float64, 2})

        @assert size(oe_integrals_tensor)[1] == size(oe_integrals_tensor)[2]
        oe_mpo = new(Array{MPO, 1}())
        site_count = size(oe_integrals_tensor)[1]

        oe_operator_mpo = OneElectronOperatorMPO(site_count)
        oe_mpo.mpos = to_mpos(oe_operator_mpo, oe_integrals_tensor)

        return oe_mpo

    end
end

function Base.show(io::IO, oe_mpo::OneElectronMPO)
    site_count = oe_mpo.mpos[1].site_count
    println(io, "$site_count-site OneElectronMPO with $(length(oe_mpo.mpos)) MPOs")
end

function to_matrix(oe_mpo::OneElectronMPO)
    sum(to_matrix.(oe_mpo.mpos))
end

Base.getindex(oe_mpo::OneElectronMPO, mpo_index::Int) = oe_mpo.mpos[mpo_index]
Base.lastindex(oe_mpo::OneElectronMPO) = length(oe_mpo.mpos)

function Base.size(oe_mpo::OneElectronMPO)
    [size(mpo) for mpo in oe_mpo.mpos]
end
