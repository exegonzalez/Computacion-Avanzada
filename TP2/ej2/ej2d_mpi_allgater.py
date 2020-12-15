#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Creo un vector con la logitud de la cantidad de procesos
x = np.arange(size, dtype=np.int) + rank

print(f'Soy el Proceso {rank}, mi vector es {x}')

# Creo la matriz donde voy a almacenar el resultado del all gather
y = np.zeros((size, len(x)), dtype=np.int)

# Comunicacion colectiva
comm.Allgather([x, MPI.INT], [y, MPI.INT])

print(f'Soy el Proceso {rank}, y recibo {y}')