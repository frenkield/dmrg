import ITensors
using LinearAlgebra
include("fcidump.jl")
"""
    itensor_tensor_to_array(tensor::ITensors.ITensor)

Convert an ITensor to a normal Julia multidimensional array.
"""
function itensor_tensor_to_array(tensor::ITensors.ITensor)

    if length(tensor.inds) == 2
        i,j = tensor.inds
        dimensions = (ITensors.dim(i), ITensors.dim(j))
        return [tensor[i=>c[1], j=>c[2]] for c in CartesianIndices(dimensions)]

    elseif length(tensor.inds) == 3
        i,j,k = tensor.inds
        dimensions = (ITensors.dim(i), ITensors.dim(j), ITensors.dim(k))
        return [tensor[i=>c[1], j=>c[2], k=>c[3]] for c in CartesianIndices(dimensions)]

    else
        i,j,k,l = tensor.inds
        dimensions = (ITensors.dim(i), ITensors.dim(j), ITensors.dim(k), ITensors.dim(l))
        return [tensor[i=>c[1], j=>c[2], k=>c[3], l=>c[4]] for c in CartesianIndices(dimensions)]
    end

end

"""
    create_heisenberg_mpo(site_count::Int)

Create an ITensor.MPO that represents the hamiltonian for a Heisenberg spin chain.
"""
function create_heisenberg_mpo(site_count::Int)

    sites = ITensors.siteinds("S=1/2", site_count)

    auto_mpo = ITensors.AutoMPO()

    for j = 1:site_count-1
        auto_mpo += "Sz", j, "Sz", j+1
        auto_mpo += 0.5, "S+", j, "S-", j+1
        auto_mpo += 0.5, "S-", j, "S+", j+1
    end

    itensor_mpo = ITensors.MPO(auto_mpo, sites)
    return itensor_mpo

end

"""
    create_one_electron_mpo(sites::Array{ITensors.Index{Int64}, 1},
                            one_electron_tensor::Array{Float64, 2})

Create an ITensor.MPO that represents the hamiltonian for the one-electron
part of the molecular hamiltonian.
"""
function create_one_electron_mpo(sites::Array{ITensors.Index{Int64}, 1},
                                 one_electron_tensor::Array{Float64, 2})

    site_count = length(sites)
    mpo = ITensors.AutoMPO()

    for c in CartesianIndices(size(one_electron_tensor))
        t = one_electron_tensor[c]
        mpo += t, "Cdagup", c[1], "Cup", c[2]
        mpo += t, "Cdagdn", c[1], "Cdn", c[2]
    end

    return ITensors.MPO(mpo, sites)

end

"""
    create_one_electron_mpo(sites::Array{ITensors.Index{Int64}, 1},
                            one_electron_tensor::Array{Float64, 2})

Create an ITensor.MPO that represents the hamiltonian for the two-electron
part of the molecular hamiltonian.
"""
function create_two_electron_mpo(sites::Array{ITensors.Index{Int64},1},
                                         two_electron_tensor::Array{Float64, 4})

    site_count = size(sites)
    mpo = ITensors.AutoMPO()

    for c in CartesianIndices(size(two_electron_tensor))
        u = 1/2 * two_electron_tensor[c]
        mpo += u, "Cdagup", c[1], "Cdagup", c[2], "Cup", c[3], "Cup", c[4]
        mpo += u, "Cdagup", c[1], "Cdagdn", c[2], "Cdn", c[3], "Cup", c[4]
        mpo += u, "Cdagdn", c[1], "Cdagup", c[2], "Cup", c[3], "Cdn", c[4]
        mpo += u, "Cdagdn", c[1], "Cdagdn", c[2], "Cdn", c[3], "Cdn", c[4]
    end

    return ITensors.MPO(mpo, sites)

end

"""
function molecular_dmrg(one_electron_tensor::Array{Float64, 2},
                        two_electron_tensor::Array{Float64, 4};
                        max_bond_dimension=500, sweep_count=10)

Create an ITensor.MPO that represents the hamiltonian for the two-electron
part of the molecular hamiltonian.
"""
function molecular_dmrg(one_electron_tensor::Array{Float64, 2}, two_electron_tensor::Array{Float64, 4};
                        max_bond_dimension=500, sweep_count=10)

    site_count = size(one_electron_tensor)[1]
    sites = ITensors.siteinds("Electron", site_count)

    one_electron_mpo = create_one_electron_mpo(sites, one_electron_tensor)
    two_electron_mpo = create_two_electron_mpo(sites, two_electron_tensor)

    random_mps = ITensors.randomMPS(sites)

    sweeps = ITensors.Sweeps(sweep_count)
    ITensors.maxdim!(sweeps, [max_bond_dimension for i in 1:length(sweeps)]...)
    ITensors.cutoff!(sweeps, 1e-10)

    energy, state = ITensors.dmrg([one_electron_mpo, two_electron_mpo], random_mps, sweeps)
    return energy, state
end

"""
function molecular_dmrg(one_electron_tensor::Array{Float64, 2},
                        two_electron_tensor::Array{Float64, 4};
                        max_bond_dimension=500, sweep_count=10)

Create an ITensor.MPO that represents the hamiltonian for the two-electron
part of the molecular hamiltonian.
"""
function molecular_dmrg(fcidump_filename::String; max_bond_dimension=500, sweep_count=10)
    t,u = read_electron_integral_tensors(fcidump_filename)
    return molecular_dmrg(t, u; max_bond_dimension, sweep_count)
end
