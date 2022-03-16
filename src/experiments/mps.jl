# https://arxiv.org/pdf/1911.11566.pdf

using LinearAlgebra

function makeMPS(vect, physInd, N)

    mps = []

    M = reshape(vect, physInd, physInd^(N-1))

    Lindsize = 1

    for i = 1:N-1

        println("----- $(size(M))")

        U, D, V = svd(M)
        temp = reshape(U, Lindsize, physInd, size(D, 1))

        push!(mps, temp)
        Lindsize = size(D, 1)
        DV = Diagonal(D) * V'

        if i == N-1
            temp = reshape(DV, Lindsize[1], physInd, 1)
            push!(mps, temp)

        else
            Rsize = cld(size(M, 2), physInd)
            M = reshape(DV, size(D, 1) * physInd, Rsize)
        end
    end

    return mps
end

N = 6
physInd = 2

vect = rand(ComplexF64, physInd^N, 1)
vect /= norm(vect)

mps = makeMPS(vect, physInd, N)

@show length(mps)

#for m in mps
#    println("=================")
#    display(mps[1])
#end