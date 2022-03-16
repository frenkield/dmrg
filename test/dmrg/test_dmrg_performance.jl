using Test
using Profile
include("../../src/dmrg/dmrg.jl")
include("../../src/dmrg/operator_mpo.jl")
include("../../src/dmrg/hamiltonian/heisenberg_hamiltonian.jl")




Profile.clear()

mpo = HeisenbergHamiltonian(40).tensor_mpo
mps = random_mps(2, mpo.site_count)

dmrg = DMRG(mpo, mps)

@profile compute_ground_state!(dmrg)
Profile.print(sortedby=:count, format=:flat)



















#=

mpo = HeisenbergHamiltonian(50).tensor_mpo
mps = random_mps(2, mpo.site_count)

dmrg = DMRG(mpo, mps)

Profile.clear()

@profile compute_ground_state!(dmrg)

Profile.print(sortedby=:count, format=:flat)



# energy = compute_ground_state!(dmrg)
# @show energy


println()





# a = randn(Tuple([2 for i in 1:mpo.site_count]))




site_count = 10
mpo = HeisenbergHamiltonian(site_count).tensor_mpo

@time @profile begin

    let
        for i in 1:5

            a = randn([2 for i in 1:site_count]...)
            mps = MPS(a)
            dmrg = DMRG(mpo, mps)

            compute_ground_state!(dmrg)

        end
    end
end

Profile.print(sortedby=:count, format=:flat)



# set_zero_subnormals(true)
# @fastmath @inbounds begin
# @test isapprox(energy, -3.73632170, atol=0.0000001)
# @test isapprox(energy, -4.25803520, atol=0.0000001)

=#