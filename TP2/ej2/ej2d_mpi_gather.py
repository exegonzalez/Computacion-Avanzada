#!/usr/bin/env python3

from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Creo un numero random
data = random.randint(0, 10)
print(f'Soy el proceso, {rank}, y mi numero es, {data}')

# Comunicacion colectiva
buffer = comm.gather(data, root=0)

if(rank == 0):
    # El proceso 0 es el que recibe los datas
    print(f'Soy el proceso {rank} y mis datos son, {buffer}')
