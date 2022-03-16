include("../../stong_basis.jl")
include("../../../experiments/units.jl")

nuclear_distance_in_bohrs = angstroms_to_bohrs(0.74)
half_nuclear_distance_in_bohrs = nuclear_distance_in_bohrs / 2

nucleus_1_position = [-half_nuclear_distance_in_bohrs, 0, 0]
nucleus_2_position = [half_nuclear_distance_in_bohrs, 0, 0]






function atomic_orbital_1_not_scaled_3d(x::AbstractArray{Float64, 1})
    sto3g_not_scaled_3d(x, nucleus_1_position)
end

function atomic_orbital_2_not_scaled_3d(x::AbstractArray{Float64, 1})
    sto3g_not_scaled_3d(x, nucleus_2_position)
end


function atomic_orbital_1_scaled_3d(x::AbstractArray{Float64, 1})
    sto3g_scaled_3d(x, nucleus_1_position)
end

function atomic_orbital_2_scaled_3d(x::AbstractArray{Float64, 1})
    sto3g_scaled_3d(x, nucleus_2_position)
end


# =======================================================
# molecular orbitals not scaled
# =======================================================

function molecular_orbital_symmetric_not_scaled_3d(x::AbstractArray{Float64, 1})
    atomic_orbital_1_not_scaled_3d(x) + atomic_orbital_2_not_scaled_3d(x)
end

function molecular_orbital_antisymmetric_not_scaled_3d(x::AbstractArray{Float64, 1})
    atomic_orbital_1_not_scaled_3d(x) - atomic_orbital_2_not_scaled_3d(x)
end

# =======================================================
# molecular orbitals scaled
# =======================================================

function molecular_orbital_symmetric_scaled_3d(x::AbstractArray{Float64, 1})
    atomic_orbital_1_scaled_3d(x) + atomic_orbital_2_scaled_3d(x)
end

function molecular_orbital_antisymmetric_scaled_3d(x::AbstractArray{Float64, 1})
    atomic_orbital_1_scaled_3d(x) - atomic_orbital_2_scaled_3d(x)
end


