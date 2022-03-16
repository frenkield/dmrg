using SymPy
using Parameters

abstract type QuantumSystem end

@with_kw struct SpinSystem <: QuantumSystem

    space_dimension = 2

    _up = [1.0; 0]
    _down = [0.0; 1]

    _zero = [0.0 0; 0 0]

    _spin_z = [0.5 0; 0 -0.5]
    _spin_x = [0 0.5; 0.5 0]
    _spin_y = [0 -im * 0.5; im * 0.5 0]

    _spin_plus = real(_spin_x + im * _spin_y)
    _spin_minus = real(_spin_x - im * _spin_y)

    _id = [1.0 0; 0 1]

    # jordan-wigner operator
    _id_jw = [1.0 0; 0 -1]

    _number = [0.0 0; 0 1]

    # ===============================================

    up = sympy.Symbol("↑", commutative=False)
    down = sympy.Symbol("↓", commutative=False)

    zero = sympy.Symbol("0", commutative=False)
    spin_z = sympy.Symbol("S_z", commutative=False)
    spin_x = sympy.Symbol("S_x", commutative=False)
    spin_y = sympy.Symbol("S_y", commutative=False)
    spin_plus = sympy.Symbol("S_+", commutative=False)
    spin_minus = sympy.Symbol("S_-", commutative=False)
    id = sympy.Symbol("I", commutative=False)
    id_jw = sympy.Symbol("I_jw", commutative=False)
    number = sympy.Symbol("η", commutative=False)

    values = Dict{Sym, SymMatrix}(
        up => SymMatrix(_up),
        down => SymMatrix(_down),
        zero => SymMatrix(_zero),
        spin_z => SymMatrix(_spin_z),
        spin_x => SymMatrix(_spin_x),
        spin_y => SymMatrix(_spin_y),
        spin_plus => SymMatrix(_spin_plus),
        spin_minus => SymMatrix(_spin_minus),
        id => SymMatrix(_id),
        id_jw => SymMatrix(_id_jw),
        number => SymMatrix(_number)
    )

end

# TODO - ce truc va peut-etre abimer la performance
function Base.getindex(spin_system::SpinSystem, symbol::Sym)

    value = spin_system.values[symbol].tolist()

    if length(size(s.values[s.up].tolist())) == 1
        return value
    else
        return Float64.(N(value))
    end
end
