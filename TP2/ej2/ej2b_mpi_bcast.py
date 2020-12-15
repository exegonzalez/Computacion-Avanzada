#!/usr/bin/env python3

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
transmitter = 0

if rank == transmitter:
    buffer = input(f'Hola, soy el proceso {rank}, ingrese un número: ')

    for i in range(size):
        req = comm.isend(buffer, dest=i, tag=i)
    req = comm.irecv(source=transmitter, tag=rank)
    data = req.wait()
    print(f'Hola, soy el proceso {rank} y el número es el {data}')
else:
    req = comm.irecv(source=transmitter, tag=rank)
    data = req.wait()
    print(f'Hola, soy el proceso {rank} y el número es el {data}')

