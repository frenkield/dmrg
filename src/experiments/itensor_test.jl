using ITensors



function create_hamiltonian(sites, N)

    ampo = AutoMPO()

    for j = 1:N-1
        ampo += "Sz", j, "Sz", j+1
        ampo += 0.5, "S+", j, "S-", j+1
        ampo += 0.5, "S-", j, "S+", j+1
    end

    H = MPO(ampo, sites)

    return H

end




function test_itensor_dmrg(N)

    sites = siteinds("S=1/2", N)

    H = create_hamiltonian(sites, N)

    psi0 = randomMPS(sites)

    sweeps = Sweeps(5)
    maxdim!(sweeps, 10, 20, 100, 100, 200)
    cutoff!(sweeps, 1e-10)

    @show sweeps

    energy, psi = dmrg(H, psi0, sweeps)

    @show psi

    #println("Final energy = $energy")

    return psi
end

function tensor_from_mps(mps)

    tensor = mps[1]

    for i in 2:length(mps)
        tensor *= mps[i]
    end

    return tensor

end

# spin_count = 3
#
# sites = siteinds("S=1/2", spin_count)
#
# # state = [isodd(n) ? "Up" : "Dn" for n in 1:spin_count]
#
# state = ["Up" "Up" "Up"] # + ["Dn" "Dn" "Dn"]
#
# psi = productMPS(sites, state)





function create_simple_hamiltonian(sites, operator, site)

    mpo_builder = AutoMPO()
    mpo_builder += operator, site
    H = MPO(mpo_builder, sites, exact=true)

    return H

end


# N = 2
#
# sites = siteinds("Electron", N)

# creation_up_1 = create_simple_hamiltonian(sites, "Adagup", 1)
# creation_up_2 = create_simple_hamiltonian(sites, "Adagup", 2)
# creation_down_1 = create_simple_hamiltonian(sites, "Adagdn", 1)
# creation_down_2 = create_simple_hamiltonian(sites, "Adagdn", 2)
#
# annihilation_up_1 = create_simple_hamiltonian(sites, "Aup", 1)
# annihilation_up_2 = create_simple_hamiltonian(sites, "Aup", 2)
# annihilation_down_1 = create_simple_hamiltonian(sites, "Adn", 1)
# annihilation_down_2 = create_simple_hamiltonian(sites, "Adn", 2)


#H = create_fci_hamiltonian(sites, N)

# state_configuration = ["Up" for i in 1:N]
# up_state = productMPS(sites, state_configuration)


#@show tensor_from_mps(empty)


#one_up = H * empty
#@show tensor_from_mps(one_up)

# creation_up = "Adagup"
# annihilation_up = "Aup"

# #let
#
#     mpo_builder = AutoMPO()
#
#     mpo_builder += creation_up, 2, annihilation_up, 1
#     mpo_builder += -1, annihilation_up, 1, creation_up, 2
#
# #    mpo_builder += "Adagup", 2, "Aup", 1
# #    mpo_builder += -1, "Aup", 1, "Adagup", 2
#
#     H = MPO(mpo_builder, sites)
#
#     state = H * empty_state
#
#     @show tensor_from_mps(empty_state)
#
#     @show tensor_from_mps(state)
#
#     norm(state)
#
#
# #end


function sonder_mps(N)

    sites = siteinds("S=1/2", N)

    up_state = productMPS(sites, "Up")
    down_state = productMPS(sites, "Dn")

    up_down_state = up_state + down_state
    up_state_full = up_down_state + -down_state

    tensors = [array(t) for t in up_state_full]


    return up_state_full, tensors

end