#!/usr/bin/env python3

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank != 0:
    data = f'Hola desde el proceso {rank}'
    req = comm.isend(data, dest=0, tag=rank)
    req.wait()
else:
    print(f'Hola soy el proceso {rank} (hay {size} procesos) y recibo:')
    for i in range(1, size):
        req = comm.irecv(source=i, tag=i)
        data = req.wait()
        print(f'{data}')
