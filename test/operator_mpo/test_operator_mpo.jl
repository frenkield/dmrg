using Test

include("../../src/operator_mpo/operator_mpo.jl")
include("../../src/operator_mpo/spin_system.jl")

#==
 = These DMRG tests specifically use OperatorMPO to generate MPOs.
 = The OperatorMPO is an implementation of the MPO construction method
 = described in Snajberk.
 =
 = In dmrg/test_dmrg.jl we instead use ITensor to construct MPOs.
 = Needless to say, the ITensor method is vastly superior.
 =
 = Long story short, all the OperatorMPO code (including this test) isn't particularly
 = useful.
 =#

function verify_display(value, expected)
    io = IOBuffer();
    show(io, value)
    @test String(take!(io)) == expected
end

s = SpinSystem()
@unpack id, spin_z, spin_x, spin_y, number = s
zed = s.zero
@vars ϵ_1 ϵ_2 ϵ_3

# =============================================================

mpo = OperatorMPO(2)

a = [1 s.spin_z]
add_operator_matrix!(mpo, [1 s.spin_z])
add_operator_matrix!(mpo, [1; s.spin_z])

mpo = OperatorMPO(2)
add_operator_matrix!(mpo, [1 s.spin_z])
@test_throws AssertionError add_operator_matrix!(mpo, [1 s.spin_z; s.spin_x s.spin_y])

a = [1 s.spin_z; s.spin_x s.spin_y; 1 s.spin_z; s.spin_x s.spin_y]
a = reshape(a, (2, 2, 2))
@test_throws AssertionError add_operator_matrix!(mpo, a)

a = [1 s.spin_z; s.spin_x s.spin_y]
@test_throws AssertionError add_operator_matrix!(mpo, a)

mpo = OperatorMPO(2)
add_operator_matrix!(mpo, [1 s.spin_z])
add_operator_matrix!(mpo, [1; s.spin_z])

io = IOBuffer();
product = sympy.MatMul(mpo.matrices[1], mpo.matrices[2], evaluate = False)
println(io, product)
@test String(take!(io)) == "Matrix([[1, S_z]])*Matrix([\n[  1],\n[S_z]])\n"

io = IOBuffer();
text_display = TextDisplay(io)
display(text_display, product)
@test String(take!(io)) == "         ⎡ 1 ⎤\n[1  S_z]⋅⎢   ⎥\n         ⎣S_z⎦"

mpo = OperatorMPO(3)
add_operator_matrix!(mpo, [1 s.spin_z])
add_operator_matrix!(mpo, [1 s.spin_z; s.spin_z 1])
add_operator_matrix!(mpo, [1; s.spin_z])

product = sympy.KroneckerProduct(mpo.matrices[1], mpo.matrices[2])
product = sympy.KroneckerProduct(product, mpo.matrices[3])

io = IOBuffer();
show(io, MIME"text/latex"(), mpo)

@test String(take!(io)) ==
    "\$\\begin{equation*}\\left(\\left[\\begin{matrix}1 & S_{z}\\end{matrix}\\right] " *
    "\\otimes \\left[\\begin{matrix}1 & S_{z}\\\\S_{z} & 1\\end{matrix}\\right]\\right) " *
    "\\otimes \\left[\\begin{matrix}1\\\\S_{z}\\end{matrix}\\right]\\end{equation*}\$\n"

# ==========================================================

mpo = OperatorMPO(2)
add_operator_matrix!(mpo, [s.id s.spin_z])
add_operator_matrix!(mpo, [s.spin_z; s.id])

full_matrix = to_matrix(mpo, s);
@test full_matrix == SymMatrix([1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1])
@test full_matrix == SymMatrix(kron(s._id, s._spin_z) + kron(s._spin_z, s._id))

# ==========================================================

mpo = OperatorMPO(3)
add_operator_matrix!(mpo, [s.id s.spin_z])
add_operator_matrix!(mpo, [s.spin_x s.spin_y; s.spin_y s.spin_z])
add_operator_matrix!(mpo, [s.spin_z; s.id])

full_matrix = to_matrix(mpo, s);

expected = kron(s._id, s._spin_x, s._spin_z) +
           kron(s._id, s._spin_y, s._id) +
           kron(s._spin_z, s._spin_y, s._spin_z) +
           kron(s._spin_z, s._spin_z, s._id)

@test full_matrix == SymMatrix(expected)


mpo = OperatorMPO(2)
add_operator_matrix!(mpo, [id ϵ_1 * number])
add_operator_matrix!(mpo, [ϵ_2 * number; id])
verify_display(prod(mpo), "ϵ_1*η*I + ϵ_2*I*η")

mpo = OperatorMPO(3)
add_operator_matrix!(mpo, [id ϵ_1 * number])
add_operator_matrix!(mpo, [id ϵ_2 * number; 0 id])
add_operator_matrix!(mpo, [ϵ_3 * number; id])
verify_display(prod(mpo), "ϵ_1*η*I^2 + ϵ_2*I*η*I + ϵ_3*I^2*η")

# ==========================================================

mpo = OperatorMPO(2)
add_operator_matrix!(mpo, [id 2*spin_z])
add_operator_matrix!(mpo, [spin_z; id])

full_matrix = to_matrix(mpo, s)
@test full_matrix == SymMatrix(kron(s._id, s._spin_z) + 2 * kron(s._spin_z, s._id))

mpo = OperatorMPO(3)
add_operator_matrix!(mpo, [id 2*number])
add_operator_matrix!(mpo, [id 3*number; 0*zed id])
add_operator_matrix!(mpo, [4 * number; id])

full_matrix = to_matrix(mpo, s);

expected = [
    0 0 0 0 0 0 0 0;
    0 4.0 0 0 0 0 0 0;
    0 0 3.0 0 0 0 0 0;
    0 0 0 7.0 0 0 0 0;
    0 0 0 0 2.0 0 0 0;
    0 0 0 0 0 6.0 0 0;
    0 0 0 0 0 0 5.0 0;
    0 0 0 0 0 0 0 9.0
]

@test full_matrix == SymMatrix(expected)

# ==========================================================

mpo = OperatorMPO(2)
add_operator_matrix!(mpo, [id 2spin_z])
add_operator_matrix!(mpo, [spin_z; id])

sum_mpo = mpo + mpo
@test sum_mpo[1] == [id 2spin_z id 2spin_z]
@test sum_mpo[2] == [spin_z; id; spin_z; id]

mpo = OperatorMPO(3)
add_operator_matrix!(mpo, [id 2number])
add_operator_matrix!(mpo, [id 3number; 0zed id])
add_operator_matrix!(mpo, [4number; id])

sum_mpo = mpo + mpo
@test sum_mpo[1] == [id 2number id 2number]
@test sum_mpo[2] == [id 3number 0zed 0zed; 0zed id 0zed 0zed;
                     0zed 0zed id 3number; 0zed 0zed 0zed id]
@test sum_mpo[3] == [4number; id; 4number; id]
