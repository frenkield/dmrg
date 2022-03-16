using LinearAlgebra
using HCubature

function norm_3d(f)
    f_squared_3d(r::Float64) = f(r)^2 * r^2
    value,error = hquadrature(r -> f_squared_3d(r), 0, 10)
    return 4 * pi * value
end

function overlap_3d(f, g)
    overlap(r::Float64) = f(r) * g(r) * r^2
    value,error = hquadrature(r -> overlap(r), 0, 10)
    return 4 * pi * value
end

struct STO3GHydrogen

    coefficients::Array{Float64, 1}
    exponents::Array{Float64, 1}
    normalizations::Array{Float64, 1}
    normed_coefficients::Array{Float64, 1}

    function STO3GHydrogen()
        exponents = [3.425250914, 0.6239137298, 0.1688554040]
        coefficients = [0.1543289673, 0.5353281423, 0.4446345422]
        normalizations = [(2*e/pi)^0.75 for e in exponents]
        normed_coefficients = coefficients .* normalizations
        new(coefficients, exponents, normalizations, normed_coefficients)
    end

end

function atomic_orbital(basis::STO3GHydrogen, center::Array{Float64, 1})

    function ao(x::AbstractArray{Float64, 1})

        position_squared = sum((x - center) .^ 2)

        return basis.normed_coefficients[1] * exp(-basis.exponents[1] * position_squared) +
               basis.normed_coefficients[2] * exp(-basis.exponents[2] * position_squared) +
               basis.normed_coefficients[3] * exp(-basis.exponents[3] * position_squared)
    end

    return ao

end

function sto1g(x::AbstractArray{Float64, 1})
    alpha1 = 0.28294212
    coefficient = (2 * alpha1 / pi)^0.75
    coefficient * exp(-alpha1 * sum(x.^2))
end

function sto2g(x::AbstractArray{Float64, 1})

    a1 = 1.309756377
    c1 = 0.4301284983

    a2 = 0.2331359749
    c2 = 0.6789135305

    x_sq = sum(x .^ 2)

    return c1 * (2 * a1 / pi)^0.75 * exp(-a1 * x_sq) +
           c2 * (2 * a2 / pi)^0.75 * exp(-a2 * x_sq)
end

function sto3g(x::AbstractArray{Float64, 1})

    a1 = 3.425250914
    c1 = 0.1543289673
    n1 = (2 * a1 / pi)^0.75 # 1.7944418337900938

    a2 = 0.6239137298
    c2 = 0.5353281423
    n2 = (2 * a2 / pi)^0.75 # 0.5003264922111158

    a3 = 0.1688554040
    c3 = 0.4446345422
    n3 = (2 * a3 / pi)^0.75 # 0.1877354618463613

    x_sq = sum(x .^ 2)

    return c1 * n1 * exp(-a1 * x_sq) +
           c2 * n2 * exp(-a2 * x_sq) +
           c3 * n3 * exp(-a3 * x_sq)

end

function slater_1s(x::AbstractArray{Float64, 1})
    sqrt_pi_inverse * exp(-norm(x))
end

# http://www.phys.ubbcluj.ro/~vasile.chis/cursuri/cspm/course6.pdf

function sto3g_scaled_3d(x::AbstractArray{Float64, 1}, center = [0.0, 0, 0])
    x_sq = sum((x - center) .^ 2)
    0.276934 * exp(-3.425250 * x_sq) +
    0.267839 * exp(-0.623913 * x_sq) +
    0.083474 * exp(-0.168856 * x_sq)
end

function sto3g_not_scaled_3d(x::AbstractArray{Float64, 1}, center = [0.0, 0, 0])
    x_sq = sum((x - center) .^ 2)
    0.200560 * exp(-2.227660 * x_sq) +
    0.193973 * exp(-0.405771 * x_sq) +
    0.060453 * exp(-0.109818 * x_sq)
end

function sto3g_scaled(r::Float64, center = 0.0)
    r_sq = (r - center)^2
    0.276934 * exp(-3.425250 * r_sq) +
    0.267839 * exp(-0.623913 * r_sq) +
    0.083474 * exp(-0.168856 * r_sq)
end

function sto3g_not_scaled(r::Float64, center = 0.0)
    r_sq = (r - center)^2
    0.200560 * exp(-2.227660 * r_sq) +
    0.193973 * exp(-0.405771 * r_sq) +
    0.060453 * exp(-0.109818 * r_sq)
end
