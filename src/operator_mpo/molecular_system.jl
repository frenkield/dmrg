using SymPy
using Parameters

abstract type QuantumSystem end

@with_kw struct MolecularSystem <: QuantumSystem

    space_dimension = 4

    _vacuum = [1.0; 0; 0; 0]
    _up = [0.0; 1; 0; 0]
    _down = [0.0; 0; 1; 0]
    _both = [0.0; 0; 0; 1]

    _zero = [0.0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]
    _id = [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

    _c_up = [0 0 0 0; 0 0 0 0; 1 0 0 0; 0 1 0 0]
    _a_up = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]

    _c_down = [0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 -1 0]
    _a_down = [0 1 0 0; 0 0 0 0; 0 0 0 -1; 0 0 0 0]

    _number = _c_up * _a_up + _c_down * _a_down

    _zc_up_left =    [0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 -1 0]
    _za_up_left =    [0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
    _zc_down_left =  [0 0 0 0; 0 0 0 0; 1 0 0 0; 0 1 0 0]
    _za_down_left =  [0 0 1 0; 0 0 0 -1; 0 0 0 0; 0 0 0 0]
    _zc_up_right =   [0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
    _za_up_right =   [0 1 0 0; 0 0 0 0; 0 0 0 -1; 0 0 0 0]
    _zc_down_right = [0 0 0 0; 0 0 0 0; 1 0 0 0; 0 -1 0 0]
    _za_down_right = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]

    _f = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
    _id_jw = _f

    # ===============================================

    vacuum = Sym("vac")
    up = Sym("↑")
    down = Sym("↓")
    both = Sym("↕")

    zero = sympy.Symbol("0")
    id = sympy.Symbol("I", commutative=False)
    id_jw = sympy.Symbol("I_jw", commutative=False)

    c_up = sympy.Symbol("c_↑^†", commutative=False)
    c_down = sympy.Symbol("c_↓^†", commutative=False)
    a_up = sympy.Symbol("c_↑", commutative=False)
    a_down = sympy.Symbol("c_↓", commutative=False)
    number = sympy.Symbol("η", commutative=False)

    zc_up_left = sympy.Symbol("ζ_↑^{L^†}", commutative=False)
    za_up_left = sympy.Symbol("ζ_↑^L", commutative=False)
    zc_down_left = sympy.Symbol("ζ_↓^{L^†}", commutative=False)
    za_down_left = sympy.Symbol("ζ_↓^L", commutative=False)

    zc_up_right = sympy.Symbol("ζ_↑^{R^†}", commutative=False)
    za_up_right = sympy.Symbol("ζ_↑^R", commutative=False)
    zc_down_right = sympy.Symbol("ζ_↓^{R^†}", commutative=False)
    za_down_right = sympy.Symbol("ζ_↓^R", commutative=False)

    f = sympy.Symbol("f", commutative=False)

    values = Dict{Sym, SymMatrix}(

        vacuum => SymMatrix(_vacuum),
        up => SymMatrix(_up),
        down => _down,
        both => _both,

        zero => SymMatrix(_zero),
        id => SymMatrix(_id),
        id_jw => SymMatrix(_id_jw),

        c_up => SymMatrix(_c_up),
        c_down => SymMatrix(_c_down),
        a_up => SymMatrix(_a_up),
        a_down => SymMatrix(_a_down),
        number => SymMatrix(_number),

        zc_up_left => SymMatrix(_zc_up_left),
        za_up_left => SymMatrix(_za_up_left),
        zc_down_left => SymMatrix(_zc_down_left),
        za_down_left => SymMatrix(_za_down_left),
        zc_up_right => SymMatrix(_zc_up_right),
        za_up_right => SymMatrix(_za_up_right),
        zc_down_right => SymMatrix(_zc_down_right),
        za_down_right => SymMatrix(_za_down_right),

        f => SymMatrix(_f)
    )

end

# TODO - ce truc va peut-etre abimer la performance
function Base.getindex(molecular_system::MolecularSystem, symbol::Sym)
    Float64.(N(molecular_system.values[symbol].tolist()))
end

# sympy.KroneckerProduct(s.values[c_up], s.values[c_down])