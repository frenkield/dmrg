using Test
using LinearAlgebra
using Cubature


alpha1 = 0.28294212
c1 = 1.0
coefficient = c1 * (2 * alpha1 / pi)^0.75

sto1g(x::Array{Float64, 1}) = coefficient * exp(-alpha1 * sum(x.^2))

function sto1g_operated(x::Array{Float64, 1})

    r_sq = sum(x.^2)
    r = sqrt(r_sq)

    result = -1/2 * coefficient * alpha1 * exp(-alpha1 * r_sq) *
             (4 * alpha1 * r_sq - 3)

    result = result - 1/(r + 1e-15) * coefficient * exp(-alpha1 * r_sq)

    return result

end


normalization = hcubature(x -> sto1g(x)^2, [-10, -10, -10], [10, 10, 10])
@test isapprox(normalization[1], 1.0, atol=0.0000001)


energy = hcubature(x -> sto1g(x) * sto1g_operated(x), [-10, -10, -10], [10, 10, 10])

















# ======================================================


# xi_1(x::Vector{Float64}) = 0.4301284983 * exp(-0.1309756377e1 * (x[1]^2 + x[2]^2 + x[3]^2)) +
#                            0.6789135305 * exp(-0.2331359749 * (x[1]^2 + x[2]^2 + x[3]^2))

# hcubature(x -> orbital_squared(x), [-10, -10, -10], [-1e-5, -1e-5, -1e-5])



# hcubature(x::Vector{Float64} -> 1 / norm(x), [-10, -10, -10], [-1e-5, -1e-5, -1e-5])
# hcubature(x::Vector{Float64} -> 1 / (norm(x) + 1e-100), [-10, -10, -10], [10, 10, 10])




# sto1g(x) = (2 * 0.283 / pi)^0.75 * exp(-0.283 * sum(x .^ 2))
# hcubature(x -> sto1g(x)^2, [-10, -10, -10], [10, 10, 10])

