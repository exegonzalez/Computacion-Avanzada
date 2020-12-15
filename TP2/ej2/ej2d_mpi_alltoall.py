#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Cargo una fila random
m = np.random.randint(0, 10, size=(size, 1), dtype=np.int)

# Imprimo mi fila
print(f'\nSoy el proceso {rank} y GENERE: \n \n {m}\n')

# Vector donde se almacena la fila luego del alltoall
r = np.zeros((size, 1), dtype=np.int)

# Comunicacion colectiva
comm.Alltoall(m, r)

# Imprimo el resultado
for i in range(len(r)):
    print(f'Soy el proceso {rank} y RECIBI la fila {r[i]} del proceso {i}')
