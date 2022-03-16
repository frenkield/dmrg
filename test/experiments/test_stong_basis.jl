using Test
using HCubature
include("../../experiments/stong_basis.jl")
include("../../experiments/units.jl")

r_min = -10
r_max = 10

@test isapprox(norm_3d(sto3g_scaled), 1.0, atol=0.00001)
@test isapprox(norm_3d(sto3g_not_scaled), 1.0, atol=0.00001)

x_min_3d = [-10, -10, -10]
x_max_3d = [10, 10, 10]
basis = STO3GHydrogen()

normalization,error = hcubature(x -> sto1g(x)^2, x_min_3d, x_max_3d)
@test isapprox(normalization, 1.0, atol=0.0000001)

normalization,error = hcubature(x -> sto2g(x)^2, x_min_3d, x_max_3d)
@test isapprox(normalization, 1.0, atol=0.0000001)

normalization,error = hcubature(x -> sto3g(x)^2, x_min_3d, x_max_3d, maxevals=1000000)
@test isapprox(normalization, 1.0, atol=0.0000001)

ao = atomic_orbital(basis, [0.0, 0, 0])
@time normalization,error = hcubature(x -> ao(x)^2, x_min_3d, x_max_3d, maxevals=1000000)
@test isapprox(normalization, 1.0, atol=0.0000001)

normalization,error = hcubature(x -> slater_1s(x)^2, x_min_3d, x_max_3d, maxevals=1000000)
@test isapprox(normalization, 1.0, atol=0.0000001)

ao1 = atomic_orbital(basis, [0.0, 0, 0])
ao2 = atomic_orbital(basis, [0.74, 0, 0])


enfer_3d(x::AbstractArray{Float64, 1}) = exp(-2 * norm(x))
normalization,error = hcubature(x -> enfer_3d(x), x_min_3d, x_max_3d, maxevals=1000000)
@test isapprox(-1/(2*pi) * normalization, -1/2, atol=0.00001)

enfer_1d(x::Float64) = exp(-2 * x) * x^2
normalization,error = hquadrature(x -> enfer_1d(x), 0, 10, maxevals=1000000)
@test isapprox(-2 * normalization, -1/2, atol=0.00001)

# normalization,error = hcubature(x -> phi_not_scaled(x)^2, x_min_3d, x_max_3d, maxevals=1000000)
# @test isapprox(normalization, 1.0, atol=0.000001)



center = angstroms_to_bohrs(0.74)
position = [center, 0, 0]

sqrt2inv = 1/sqrt(2)

phi_1(x::AbstractArray{Float64, 1}) =
    0.5488414263983338 * (phi_scaled(x, position) + phi_scaled(x))

phi_2(x::AbstractArray{Float64, 1}) =
    1.212450205330508 * (phi_scaled(x, position) - phi_scaled(x))

# normalization,error = hcubature(x -> phi_2(x)^2, x_min, x_max, maxevals=50000000)



#=
function asdf(x_6d::AbstractArray{Float64, 1})

    x1 = view(x_6d, 1:3)
    x2 = view(x_6d, 4:6)

    distance = max(norm(x1 - x2), 1e-10)
    value = phi_1(x1)^2 * phi_1(x2)^2 / distance

    return value

end

# 0.67475592681348873

x_6d_min = [-10, -10, -10, -10, -10, -10]
x_6d_max = [10, 10, 10, 10, 10, 10]

# normalization,error = hcubature(x_6d -> asdf(x_6d), x_6d_min, x_6d_max, maxevals=5000000)

=#







