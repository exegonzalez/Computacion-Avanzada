#!/usr/bin/env python3

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
transmitter = 0

if rank == transmitter:
    buffer = input(f'Hola, soy el proceso {rank}, ingrese un número: ')

    for i in range(1, size):
        comm.send(buffer, dest=i, tag=i)

    print(f'Hola, soy el proceso {rank} y el número es el {buffer}')
else:
    data = comm.recv(source=transmitter, tag=rank)
    print(f'Hola, soy el proceso {rank} y el número es el {data}')


