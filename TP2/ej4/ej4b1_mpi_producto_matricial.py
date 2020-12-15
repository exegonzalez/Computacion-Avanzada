#!/usr/bin/env python3

import sys
import numpy as np
from mpi4py import MPI

def main(matrixSize, procSize):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    root = 0

    matrix = None # matriz n*n vacía
    bufferReceptor = np.zeros((matrixSize // procSize, matrixSize), dtype=np.int) # vector columna vacío
    vectorResultOfProduct = np.zeros((matrixSize, 1), dtype=np.int) # vector resultante vacío
    vector = None # vector columna vacío


    if rank == 0:
        matrix = np.random.randint(0, 10, size=(matrixSize, matrixSize), dtype=np.int) # matriz n*n
        vector = np.random.randint(0, 10, size=(matrixSize, 1), dtype=np.int) # vector columna

    # repartimos las filas de la matriz entre los procesos participantes
    comm.Scatter([matrix, MPI.INT], [bufferReceptor, MPI.INT])

    # repartimos el vector a multiplicar entre los procesos participantes
    vectorToMultiply = comm.bcast(vector, root)

    # realizamos los productos correspondientes
    vectorResult = np.matmul(bufferReceptor, vectorToMultiply)

    # enviamos el vector resultante de cada producto realizado por los demas procesos, al proceso 0
    productResult = comm.gather(vectorResult, root)

    if rank == 0:
        aux = 0
        for list1 in productResult:
            for list2 in list1:
                vectorResultOfProduct.put(aux, list2)
                aux += 1

        print(f'Hola soy el proceso {rank}, el producto {matrix} · {vector} = {vectorResultOfProduct}')


if __name__ == "__main__":
        haveArguments = len(sys.argv) > 1
        if not haveArguments:
            print("Debe ingresar como parametro la cantidad el tamaño de la matriz cuadrada")

        else:
            comm = MPI.COMM_WORLD
            matrizSize = int(sys.argv[1])
            procSize = comm.Get_size()
            if matrizSize % procSize == 0:
                main(matrizSize, procSize)
            else:
                print("La cantidad de filas de la matriz debe ser multiplo de la cantidad de procesos")

