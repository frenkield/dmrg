using Test
using GLMakie
include("../../../../experiments/molecules/h2/basis.jl")


function show_ground_state()

    r = LinRange(-4, 4, 100)

    ground_state(x::AbstractArray{Float64, 1}) =
        -0.112544 * molecular_orbital_antisymmetric_scaled_3d(x) +
        0.993647 * molecular_orbital_symmetric_scaled_3d(x)

    all_points = [ground_state([x, y, z]) for x = r, y = r, z = r]

    contour(all_points, levels=[-1.0, -0.04, 0.04, 1.0], alpha=0.05)

end

function show_orbital_antisymmetric()

    r = LinRange(-4, 4, 100)
    all_points = [molecular_orbital_antisymmetric_not_scaled_3d([x, y, z]) for x = r, y = r, z = r]

    colors = to_colormap(:balance)
    n = length(colors)
    alpha = [ones(n÷3); zeros(n-2*(n÷3)); ones(n÷3)]
    cmap_alpha = RGBAf0.(colors, alpha)

    # volume(all_points, algorithm=:absorption, colorrange=(-0.1,0.1), colormap=cmap_alpha)

    contour(all_points, levels=[-0.2, -0.1, 0.1, 0.2], alpha=0.2)

end