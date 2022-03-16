using LinearAlgebra

function kronecker_product(a, b)

    a_order = length(size(a))
    b_order = length(size(b))

    if a_order == 1
        a = reshape(a, (length(a), 1))
    end

    if b_order == 1
        b = reshape(b, (1, length(b)))
    end

    c = zeros(size(a)[1] * size(b)[1], size(a)[2] * size(b)[2])

    # @show size(c)

    for i in 1:size(a)[1]
        for j in 1:size(a)[2]
            c[1:1, 1:3] = a[i, j] * b
        end
    end

    return c

end

function test_kronecker_product_1()
    a = rand(3)
    b = rand(3)
    c = kronecker_product(a, b)
    @assert c == a * b'
end

function test_kronecker_product_2()

    a = rand(3, 3)
    b = rand(3, 3)

    c = kronecker_product(a, b);


    # @show c

    @assert 1 == 1

end

#test_kronecker_product_1()
#test_kronecker_product_2()


a = rand(3, 3)
b = rand(3, 3)
ab = kron(a, b)

x = rand(3)
y = rand(3)
xy = kron(x, y)

abxy = ab * xy

axby = kron(a*x, b*y)

@assert isapprox(abxy, axby)

