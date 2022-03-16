using Profile
include("../src/dmrg/itensor_utils.jl")

site_count = 8

sweeps = ITensors.Sweeps(5)
ITensors.maxdim!(sweeps, [100 for i in 1:length(sweeps)]...)
ITensors.cutoff!(sweeps, 1e-10)

sites = ITensors.siteinds("Electron", site_count)

random_mps = ITensors.randomMPS(sites)

mpo1 = create_one_electron_mpo(sites, rand(site_count, site_count))
mpo2 = create_two_electron_mpo(sites, rand(site_count, site_count, site_count, site_count))

Profile.clear()

@profile ITensors.dmrg([mpo1, mpo2], random_mps, sweeps)

Profile.print(sortedby=:count, format=:flat)
