#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# El proceso 0 genera los data
if(rank == 0):

    data = np.random.randint(0, 10, size)
    print(f'Soy el proceso 0 y genere los datos: {data}')

else:
    # Los demas procesos tambien tienen que tener definida la variable
    data = None

# Comunicacion colectiva
dataReceived = comm.bcast(data, root=0)

# Cada proceso imprime los data recibidos
print(f'Soy el proceso, {rank} y recibi {dataReceived}')
