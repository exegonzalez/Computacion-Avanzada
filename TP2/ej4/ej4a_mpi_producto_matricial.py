#!/usr/bin/env python3

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

matrix = None # matriz n*n vacía
bufferReceptor = np.zeros((1, size), dtype=np.int) # vector columna vacío
vectorResultOfProduct = np.zeros((size, 1), dtype=np.int) # vector resultante vacío
vector = None # vector columna vacío


if rank == 0:
    matrix = np.random.randint(0, 10, size=(size, size), dtype=np.int) # matriz n*n
    vector = np.random.randint(0, 10, size=(size, 1), dtype=np.int) # vector columna

# repartimos las filas de la matriz entre los procesos participantes
comm.Scatter([matrix, MPI.INT], [bufferReceptor, MPI.INT])

# repartimos el vector a multiplicar entre los procesos participantes
vectorToMultiply = comm.bcast(vector, root)

# realizamos los productos correspondientes
vectorResult = np.matmul(bufferReceptor, vectorToMultiply)

# enviamos el vector resultante de cada producto realizado por los demas procesos, al proceso 0
productResult = comm.gather(vectorResult[0], root)

if rank == 0:
    for i in range(len(productResult)):
        vectorResultOfProduct.put(i, productResult[i].item())

    print(f'Hola soy el proceso {rank}, el producto {matrix} · {vector} = {vectorResultOfProduct}')
