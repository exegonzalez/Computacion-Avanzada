#!/usr/bin/env python3

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank != 0:
    data = f'Hola desde el proceso {rank}'
    comm.send(data, dest=0, tag=rank)
else:
    print(f'Hola soy el proceso {rank} (hay {size} procesos) y recibo:')
    for i in range(1, size):
        data = comm.recv(source=i, tag=i)
        print(f'{data}')