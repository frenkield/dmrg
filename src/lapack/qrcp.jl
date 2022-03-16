using LinearAlgebra
using Test

import ..LinearAlgebra.BLAS.@blasfunc
import ..LinearAlgebra: BlasInt

function qrcp!(matrix::AbstractMatrix{Float64}, reflectors::AbstractVector{Float64},
               factorized_column_count::Int = size(matrix)[2])

    @assert length(reflectors) == factorized_column_count
    @assert length(reflectors) <= minimum(size(matrix))

    M = size(matrix)[1]
    N = size(matrix)[2]
    OFFSET = 0
    NB = factorized_column_count
    KB = Array{Int}(undef, 1)
    LDA = M
    JPVT = Array{Int}(undef, N)
    VN1 = Array{Float64}(undef, N)
    VN2 = Array{Float64}(undef, N)
    AUXV = Array{Float64}(undef, NB)
    F = Array{Float64}(undef, N * M)
    LDF = N

    for i in 1:N
        VN1[i] = norm(matrix[:, i])
        VN2[i] = VN1[i]
    end

    #==
     = overwrite matrix with combined q/r matrix
     = http://www.netlib.org/lapack/lapack-3.1.1/html/dlaqps.f.html
     =#
    ccall((@blasfunc(dlaqps_), Base.liblapack_name), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Int},
           Ptr{Float64}, Ref{BlasInt}, Ptr{Int}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}),
          M, N, OFFSET, NB, KB,
          matrix, LDA, JPVT, reflectors, VN1,
          VN2, AUXV, F, LDF)

    #==
     = extract r matrix
     = https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LAPACK-functions
     =#
    r = Array{Float64}(undef, size(matrix)[1], factorized_column_count)
    [r[i, j] = i <= j ? matrix[i, j] : 0 for i in 1:size(r)[1], j in 1:size(r)[2]]

    #==
     = overwrite matrix with just q matrix
     = https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.LAPACK.orgqr!
     =#
    LAPACK.orgqr!(matrix, v)

    return matrix, r

end

# =====================================================

# test full decomposition (extract all columns)

a = rand(3, 3)
v = zeros(3)
b = copy(a)
d = qr(b, Val(true))

q,r = qrcp!(a, v)

@test isapprox(q, d.Q, atol=0.00001)
@test isapprox(r, d.R, atol=0.00001)

# =====================================================

# extract only 1 column

a = rand(3, 3)
v = zeros(1)
b = copy(a)
d = qr(b, Val(true))

q,r = qrcp!(a, v, length(v))

@test isapprox(q[:, 1], d.Q[:, 1], atol=0.00001)
@test isapprox(r, d.R[:, 1], atol=0.00001)

# =====================================================

# extract 3 columns from bigger matrix

column_count = 3

a = rand(10, 10)
v = zeros(column_count)
b = copy(a)
d = qr(b, Val(true))

q,r = qrcp!(a, v, length(v))

@test isapprox(q[:, 1:column_count], d.Q[:, 1:column_count], atol=0.00001)
@test isapprox(r, d.R[:, 1:column_count], atol=0.00001)

# =====================================================

# extract 3 columns from wide matrix

column_count = 3

a = rand(10, 20)
v = zeros(column_count)
b = copy(a)
d = qr(b, Val(true))

q,r = qrcp!(a, v, length(v))

@test isapprox(q[:, 1:column_count], d.Q[:, 1:column_count], atol=0.00001)
@test isapprox(r, d.R[:, 1:column_count], atol=0.00001)

# =====================================================

# extract 3 columns from tall matrix

column_count = 3

a = rand(20, 10)
v = zeros(column_count)
b = copy(a)
d = qr(b, Val(true))

q,r = qrcp!(a, v, length(v))

@test isapprox(q[:, 1:column_count], d.Q[:, 1:column_count], atol=0.00001)
@test isapprox(r[1:10, :], d.R[:, 1:column_count], atol=0.00001)
