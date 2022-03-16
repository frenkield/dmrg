using SpecialFunctions
include("../../../../experiments/molecules/h2/basis.jl")

# F_0(x^2)
function boys_integral_of_square(x::Float64)
    sqrt(pi) / (2*x) * erf(x)
end

function coulomb_gaussian_integral(alpha::Float64, position1, position2)
    x = sqrt(alpha) * norm(position1 - position2)
    2*pi/alpha * boys_integral_of_square(x)
end


# =====================================================
# scaled
# =====================================================

x_3d_min = [-20, -20, -20]
x_3d_max = [20, 20, 20]

#=
integral,error = hcubature(x_3d -> atomic_orbital_1_scaled_3d(x_3d)^2, x_3d_min, x_3d_max)

@show integral

integral,error = hcubature(x_3d -> atomic_orbital_2_scaled_3d(x_3d)^2, x_3d_min, x_3d_max)
@show integral

integral,error = hcubature(x_3d -> atomic_orbital_1_scaled_3d(x_3d) * atomic_orbital_2_scaled_3d(x_3d),
                           x_3d_min, x_3d_max)

@show integral

center = [-0.7, 0, 0]

integral,error = hcubature(x_3d -> atomic_orbital_1_scaled_3d(x_3d) *
                                   atomic_orbital_1_scaled_3d(x_3d) / norm(x_3d - center),
                           x_3d_min, x_3d_max)

@show integral

=#

alpha = 1.0
center = [-0.7, 0, 0]
position = randn(3)

integral,error = hcubature(x_3d -> exp(-alpha * sum((x_3d-position) .^ 2)) /
                                   norm(x_3d - center), x_3d_min, x_3d_max)
@show integral

coulomb_gaussian_integral(alpha, center, position)





#=

integral,error = hcubature(x_3d -> molecular_orbital_symmetric_scaled_3d(x_3d)^2,
                           x_3d_min, x_3d_max)

@test isapprox(integral, 3.3197565099119117, atol=0.0000001)

function two_electron_integral_scaled_1111(x_6d::AbstractArray{Float64, 1})

    x1 = view(x_6d, 1:3)
    x2 = view(x_6d, 4:6)

    distance = max(norm(x1 - x2), 1e-13)

    value = molecular_orbital_symmetric_scaled_3d(x1)^2 *
            molecular_orbital_symmetric_scaled_3d(x2)^2 / distance

    return value

end

x_6d_min = [-20, -20, -20, -20, -20, -20]
x_6d_max = [20, 20, 20, 20, 20, 20]

integral,error = hcubature(x_6d -> two_electron_integral_scaled_1111(x_6d),
                           x_6d_min, x_6d_max, maxevals=300000000)

@show integral / 3.3197565099119117^2
@test isapprox(integral / 3.3197565099119117^2, 0.6742190294097895, atol=0.00001)









integral,error = hcubature(x_3d -> molecular_orbital_antisymmetric_scaled_3d(x_3d)^2,
                           x_3d_min, x_3d_max)

@test isapprox(integral, 0.6802556807297095, atol=0.0000001)

function two_electron_integral_scaled_2222(x_6d::AbstractArray{Float64, 1})

    x1 = view(x_6d, 1:3)
    x2 = view(x_6d, 4:6)

    distance = max(norm(x1 - x2), 1e-15)

    value = molecular_orbital_antisymmetric_scaled_3d(x1)^2 *
            molecular_orbital_antisymmetric_scaled_3d(x2)^2 / distance

    return value

end

integral,error = hcubature(x_6d -> two_electron_integral_scaled_2222(x_6d),
                           x_6d_min, x_6d_max, maxevals=300000000)

@show integral / 0.6802556807297095^2

#@test isapprox(integral / 3.3197565099119117^2, 0.6742190294097895, atol=0.00001)


=#












#=

# =====================================================
# not scaled
# =====================================================

integral,error = hcubature(x_3d -> molecular_orbital_symmetric_not_scaled_3d(x_3d)^2,
                           x_3d_min, x_3d_max)

@test isapprox(integral, 3.5069792837792995, atol=0.0000001)

function two_electron_integral_not_scaled_1111(x_6d::AbstractArray{Float64, 1})

    x1 = view(x_6d, 1:3)
    x2 = view(x_6d, 4:6)

    distance = max(norm(x1 - x2), 1e-15)

    value = molecular_orbital_symmetric_not_scaled_3d(x1)^2 *
            molecular_orbital_symmetric_not_scaled_3d(x2)^2 / distance

    return value

end

x_6d_min = [-20, -20, -20, -20, -20, -20]
x_6d_max = [20, 20, 20, 20, 20, 20]

integral,error = hcubature(x_6d -> two_electron_integral_not_scaled_1111(x_6d),
                           x_6d_min, x_6d_max, maxevals=300000000)

@show integral / 3.5069792837792995^2
#@test isapprox(integral / 3.5069792837792995^2, 0.6742190294097895, atol=0.00001)

=#
