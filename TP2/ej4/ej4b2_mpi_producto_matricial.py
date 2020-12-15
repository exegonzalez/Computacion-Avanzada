#!/usr/bin/env python3

import sys
import numpy as np
from mpi4py import MPI
from array import array

def main(matrixSize):
    """
    El inconveniente que se produce en este ejercicio es que al realizar la reparticion de las filas de la matriz en el Scatter,
    se produce un error ya que la cantidad de elementos a enviar es menor que la cantidad de elemtos a recibir, con lo cual la
    parcial encontrada fue agregar filas de 0 a la matriz para igualar la cantidad de elementos a enviar y recibir. Por lo tanto
    la reparticion de datos se realiza correctamente y el Scatter no produce fallos. Al realizar el producto matricial, estas filas
    de ceros añadidas a la matriz no afectan el mismo.
    """

    comm = MPI.COMM_WORLD
    procSize = comm.Get_size()
    rank = comm.Get_rank()
    root = 0

    # Si la cantidad de filas es multiplo de la cantidad de procesos devuelve el resto, sino devuelve el resto mas uno
    bufferReceptorSize = (matrixSize//procSize) if matrixSize%procSize == 0 else round(matrixSize//procSize)+1

    # La cantidad de filas de 0 que debe añadirse a la matriz
    numberOfRowsToAdd = (procSize * bufferReceptorSize) - matrixSize

    matrix = None # matriz n*n vacía
    bufferReceptor = np.zeros((bufferReceptorSize, matrixSize), dtype=np.int) # vector columna vacío
    vectorResultOfProduct = np.zeros((matrixSize + numberOfRowsToAdd, 1), dtype=np.int) # vector resultante vacío
    vector = None # vector columna vacío


    if rank == 0:
        # Si la cantidad de filas de 0 a añadir es multiplo de la cantidad de procesos, 
        # la diferencia sera 0, caso contrario distinto de 0
        if numberOfRowsToAdd != 0:
            matrix = np.random.randint(0, 10, size=(matrixSize, matrixSize), dtype=np.int) # matriz random de n*n
            vector = np.random.randint(0, 10, size=(matrixSize, 1), dtype=np.int) # vector columna random 

            vectorOfZeros = np.zeros((numberOfRowsToAdd, matrixSize), dtype=np.int) # vector de ceros a añadir
            matrix = np.append(matrix, np.zeros((numberOfRowsToAdd, matrixSize), dtype=np.int), 0) # añadimos el vector a la matriz
        else:
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

    # Refactorizamos la matriz resultante para mostrarla de forma adecuada
    if rank == 0:
        aux = 0
        for list1 in productResult:
            for list2 in list1:
                vectorResultOfProduct.put(aux, list2)
                aux += 1

        print(f'Hola soy el proceso {rank}, el producto {matrix} · {vector} = {vectorResultOfProduct[:-numberOfRowsToAdd]}')


if __name__ == "__main__":
        haveArguments = len(sys.argv) > 1
        if not haveArguments:
            print("Debe ingresar como parametro la cantidad el tamaño de la matriz cuadrada")

        else:
            matrizSize = int(sys.argv[1])
            main(matrizSize)
