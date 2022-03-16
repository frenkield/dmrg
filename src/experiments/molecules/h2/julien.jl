using Test

H_singlet = zeros(2,2)
H_singlet[1,1] = (-1.2533097866459868 * 2) + 0.67475592681348873 + 0.71510433908108118
H_singlet[2,2] = (-0.47506884877222921 * 2) + 0.69765150448945734 + 0.71510433908108118
H_singlet[1,2] = H_singlet[2,1] = 0.18121046201494442

H_triplet = -1.2533097866459868 + -0.47506884877222921 + 0.66371140134986573 - 0.18121046201494442 +
            0.71510433908108118

# 3-fold degenerate

# one electron
H_one = -1.2533097866459868 + 0.67475592681348873 + 0.71510433908108118
# H_one = -1.2533097866459868 + 0.67475592681348873


# 3d case - replace coulomb by delta function potential???
# dmrg + dft or perturbation theory
# hybrid solutions most interesting for julien

