"""
    read_electron_integral_tensors(fcidump_filename::String)

Read the one-electron and two-electron tensors from an FCIDUMP file.
This code was specifically written to read FCIDUMP files generated by
Quantum Package (https://quantumpackage.github.io/qp2).

Note that the two-electron is permuted at the end of the function:

    two_electron_integral_tensor = permutedims(two_electron_integral_tensor, [3,2,1,4])

It's not clear why this permutation is necessary - and it may be possible that it's
actually not necessary. However, quantum chemistry calculations using the non-permuted
two-electron tensor seem to be incorrect. Additionally, the datasets at Quantaggle
(https://github.com/qulacs/Quantaggle_dataset/tree/master/datasets/Small_Molecules_1)
don't require this permutation.

One logical explanation is that the difference is due to the ordering of
the orbitals. However, if that's the case, then the ground state energy should
be the same regardless of the permutation of orbitals. So the problem seems to lie
elsewhere.

Note that the hydrogen and water molecule calculations provided in this project
use the permuted two-electron tensor. And the results are correct - at least as far
as the author can tell.

In any case, there's some weirdness somewhere. So if you use this function for
molecules other than hydrogen and water, be sure to consult with a chemist
regarding this little mystery.
"""

function read_electron_integral_tensors(fcidump_filename::String)

    lines = readlines(fcidump_filename)
    @assert lines[4] == " /"

    orbital_count_match = match(r"NORB= *([0-9]+)", lines[1])
    orbital_count = parse(Int, orbital_count_match[1])

    one_electron_integral_tensor = zeros(orbital_count, orbital_count)

    two_electron_integral_tensor =
        zeros(orbital_count, orbital_count, orbital_count, orbital_count)

    for line in lines[5:end]

        tokens = split(line)
        value = parse(Float64, tokens[1])
        i = parse(Int, tokens[2])
        j = parse(Int, tokens[3])
        k = parse(Int, tokens[4])
        l = parse(Int, tokens[5])

        if i == j == k == l == 0

            # ENV["JULIA_DEBUG"] = "all"
            @debug begin
                println("core energy = $value")
            end

        elseif k == l == 0
            one_electron_integral_tensor[i, j] = one_electron_integral_tensor[j, i] = value

        else

            for indices in Set([(i,j,k,l), (j,i,k,l), (i,j,l,k), (j,i,l,k),
                                (k,l,i,j), (k,l,j,i), (l,k,i,j), (l,k,j,i)])

                # ENV["JULIA_DEBUG"] = "all"
                @debug begin
                    if two_electron_integral_tensor[indices...] != 0
                        @show line
                    end
                end

                two_electron_integral_tensor[indices...] = value
            end
        end

    end

    two_electron_integral_tensor = permutedims(two_electron_integral_tensor, [3,2,1,4])
    return one_electron_integral_tensor, two_electron_integral_tensor

end