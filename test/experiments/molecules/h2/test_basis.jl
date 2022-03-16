include("../../../../experiments/molecules/h2/basis.jl")

x_min_3d = [-10, -10, -10]
x_max_3d = [10, 10, 10]

@test isapprox(norm_3d(sto3g_scaled), 1.0, atol=0.00001)
@test isapprox(norm_3d(sto3g_not_scaled), 1.0, atol=0.00001)

integral,error =
    hcubature(x -> molecular_orbital_symmetric_not_scaled_3d(x)^2, x_min_3d, x_max_3d)

@test isapprox(integral, 3.5069792837792995, atol=0.0000001)

integral,error =
    hcubature(x -> molecular_orbital_antisymmetric_not_scaled_3d(x)^2, x_min_3d, x_max_3d)

@test isapprox(integral, 0.4930235042848754, atol=0.0000001)

integral,error =
    hcubature(x -> molecular_orbital_symmetric_not_scaled_3d(x) *
                   molecular_orbital_antisymmetric_not_scaled_3d(x), x_min_3d, x_max_3d)

@test isapprox(integral, 0.0, atol=0.0000001)

integral,error =
    hcubature(x -> molecular_orbital_symmetric_scaled_3d(x)^2, x_min_3d, x_max_3d)

@test isapprox(integral, 3.3197565099119117, atol=0.0000001)

integral,error =
    hcubature(x -> atomic_orbital_1_scaled_3d(x)^2, x_min_3d, x_max_3d)

@test isapprox(integral, 1.0, atol=0.0001)

integral,error =
    hcubature(x -> atomic_orbital_1_not_scaled_3d(x)^2, x_min_3d, x_max_3d)

@test isapprox(integral, 1.0, atol=0.0001)
