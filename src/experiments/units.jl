hartrees_to_evs(hartrees::Float64) = hartrees * 27.2114
evs_to_hartrees(evs::Float64) = evs / 27.2114
bohrs_to_meters(bohrs::Float64) = bohrs * 5.29177210903e-11
meters_to_bohrs(meters::Float64) = meters / 5.29177210903e-11
angstroms_to_bohrs(angstroms::Float64) = angstroms / 0.529177210903

sqrt_pi_inverse = 1 / sqrt(pi)


