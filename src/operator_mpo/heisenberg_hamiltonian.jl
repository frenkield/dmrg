include("operator_mpo.jl")
include("spin_system.jl")

struct HeisenbergHamiltonian

    site_count::Int
    operator_mpo::OperatorMPO
    tensor_mpo::MPO

    function HeisenbergHamiltonian(site_count::Int)

        s = SpinSystem()
        operator_mpo = OperatorMPO(site_count)

        add_operator_matrix!(operator_mpo,
                            [s.zero  1//2 * s.spin_minus  1//2 * s.spin_plus  s.spin_z  s.id])

        for i in 2 : site_count - 1

            add_operator_matrix!(operator_mpo,
                [s.id s.zero s.zero s.zero s.zero;
                 s.spin_plus s.zero s.zero s.zero s.zero;
                 s.spin_minus s.zero s.zero s.zero s.zero;
                 s.spin_z s.zero s.zero s.zero s.zero;
                 s.zero 1//2*s.spin_minus 1//2*s.spin_plus s.spin_z s.id])
        end

        add_operator_matrix!(operator_mpo, [s.id; s.spin_plus; s.spin_minus; s.spin_z; s.zero])

        tensor_mpo = MPO(operator_mpo, s)
        new(site_count, operator_mpo, tensor_mpo)

    end

end