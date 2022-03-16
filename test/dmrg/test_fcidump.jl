using Test
include("../../src/dmrg/fcidump.jl")

fcidump_filename = "data/h2o/h2o.ezfio.FCIDUMP"

t,u = read_electron_integral_tensors(fcidump_filename)

@test size(t) == (7, 7)
@test size(u) == (7, 7, 7, 7)

@test -6.341584396064137 == t[7,7]
@test 5.522869664237318e-6 == t[6,3] == t[3,6]

@test -0.020575263163389135 == u[6,2,3,3] == u[2,6,3,3] == u[3,3,6,2]
@test 4.823980499864563e-8 == u[7,2,3,3]

