#!/usr/bin/env python3

import numpy as np
from simplex_secuencial import simplex_init, solve_linear_program


np.random.seed(12345)
num_variables = 300
num_restricciones = 500
A = [np.random.rand(num_variables) for j in range(num_restricciones)]
c = np.random.rand(num_variables)
b = np.random.rand(num_restricciones)

base, c_p, A, b = simplex_init(c, lessThans=A, ltThreshold=b, maximization=True, M=10.)
x_opt, z_opt, _ = solve_linear_program(base, c_p, A, b)
print("La solución es:")
for j in x_opt:
  print("x_{} = {}".format(j, x_opt[j]))
print("Esto produce un funcional de z = {}".format(z_opt))