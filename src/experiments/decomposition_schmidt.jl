using ITensors

function tensor_from_mps(mps)

    tensor = mps[1]

    for i in 2:length(mps)
        tensor *= mps[i]
    end

    return tensor

end

spin_count = 3
sites = siteinds("S=1/2", spin_count)

# state = [isodd(n) ? "Up" : "Dn" for n in 1:spin_count]
# state = ["Up" "Up" "Up"] # + ["Dn" "Dn" "Dn"]

state = ["Up" "Dn" "Dn"]
psi = productMPS(sites, state)



# random_mps = randomMPS(sites, 2)

# mps = MPS(sites)

# raw_state = initState(sites, "Up")


