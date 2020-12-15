#!/usr/bin/env python3

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
receiver = 0

if rank != 0:
    comm.send(rank+1, dest=receiver, tag=rank)
else:
    factorial = 1
    for i in range(1, size):
        data = comm.recv(source=i, tag=i)
        factorial = factorial * data
    print(f'Hola soy el proceso {rank} el factorial de {size} es {factorial}')
