# ==============================================================
# https://arxiv.org/pdf/1703.04789.pdf
# ==============================================================

using LinearAlgebra
using Test

⊗(a, b) = kron(a, b)

id = I(2)

# jordan-wigner
id_t = [1 0; 0 -1]

a_dag = [0 0; 1 0]

a_dag_t = [0 0; 1 0]

a = [0 1; 0 0]

a_t = [0 -1; 0 0]

n = [0 0; 0 1]

n_t = [0 0; 0 -1]

a_1 = kron(a, id)
a_2 = kron(id_t, a)

# ============================================

K = 2
h = randn(2, 2)
V = randn(2, 2, 2, 2)

# HV_KK(p) = h_pK a_K + V_pKKK a_dag_K a_K a_K

V_op_K(p) = h[p, K] * a + V[p, K, K, K] * a_dag * a * a

P_op_K()
