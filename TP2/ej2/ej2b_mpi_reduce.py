#!/usr/bin/env python3

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
receiver = 0

if rank != 0:
    req = comm.isend(rank+1, dest=receiver, tag=rank)
    req.wait()
else:
    factorial = 1
    for i in range(1, size):
        req = comm.irecv(source=i, tag=i)
        data = req.wait()
        factorial = factorial * data
    print(f'Hola soy el proceso {rank} el factorial de {size} es {factorial}')