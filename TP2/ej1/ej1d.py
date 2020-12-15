#!/usr/bin/env python3

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
receiver = 3

if rank != receiver:
    data = f'Hola desde el proceso {rank}'
    comm.send(data, dest=receiver, tag=rank)
else:
    print(f'Hola soy el proceso {receiver} (hay {size} procesos) y recibo:')
    for i in range(0, size):
        if i != receiver:
            data = comm.recv(source=i, tag=i)
            print(f'{data}')