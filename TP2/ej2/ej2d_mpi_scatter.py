#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Tamanio del sub vector de cada proceso
miN = size // size

# Inicializo los arreglos vacios
dataSent = np.empty(size, dtype=np.int)       # SendBuff
dataReceived = np.empty(miN, dtype=np.int)  # RecvBuff

if(rank == 0):
    # El proceso 0 se encarga de cargar el arreglo random
    dataSent = np.random.randint(0, 10, size=(1, size))

    print(f'Soy el proceso 0 y los datos son {dataSent}')

# Comunicacion colectiva
comm.Scatter([dataSent, MPI.INT], [dataReceived, MPI.INT])

print(f'Soy el proceso, {rank}, y mis datos son, {dataReceived}')
