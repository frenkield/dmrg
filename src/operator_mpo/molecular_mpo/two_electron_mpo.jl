using LinearAlgebra
include("one_electron_mpo.jl")

# snajberk 5.44
mutable struct TwoElectronMPO

    site_count::Int
    molecular_system::MolecularSystem
    oe_mpos::Array{OneElectronMPO, 1}

    function TwoElectronMPO(te_integrals_tensor::Array{Float64, 4})

        @assert size(te_integrals_tensor)[1] == size(te_integrals_tensor)[2]  == size(te_integrals_tensor)[3]
        site_count = size(te_integrals_tensor)[1]

        te_mpo = new(site_count, MolecularSystem())
        te_mpo.oe_mpos = Array{OneElectronMPO, 1}()

        return te_mpo

    end
end

function add_mpos!(te_mpo::TwoElectronMPO, u::Array{Float64, 4})

    q,r = tensor_qr(u)

    u_alpha

    @tensor begin
        computed_u[m, k, n, l] := q[m, k, a] * r[a, n, l]
    end


end

function Base.show(io::IO, te_mpo::TwoElectronMPO)
    site_count = te_mpo.site_count
    println(io, "$site_count-site TwoElectronMPO")
end

# snajberk 5.41 - it seems like there's a typo - so we compute u_{mknl}
function tensor_qr(tensor::Array{Float64, 4})

    @assert size(tensor)[1] == size(tensor)[2] == size(tensor)[3] == size(tensor)[4]
    basis_dimension = size(tensor)[1]

    q,r = qr(reshape(tensor, (basis_dimension^2, basis_dimension^2)))

    q = collect(reshape(q, (basis_dimension, basis_dimension, basis_dimension^2)))
    r = collect(reshape(r, (basis_dimension^2, basis_dimension, basis_dimension)))

    return q,r

end