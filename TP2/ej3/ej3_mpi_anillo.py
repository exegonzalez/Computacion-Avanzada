#!/usr/bin/env python3

import argparse
from mpi4py import MPI

def main(n):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    message = 'A'
    
    for i in range(n):
        print(f'\nVuelta al anillo N°: {i} \n')
        
        if rank == 0:
            
            fromProc = size-1
            res = comm.irecv(None, source=fromProc)
            print(f'El proceso {rank} recibio el mensaje {message} del proceso {fromProc}', flush=True)

            nextProc = rank + 1
            comm.send(message, dest=nextProc)
            print(f'El proceso {rank} envia el mensaje {message} al proceso {nextProc}', flush=True)
            res.wait()

        else:

            fromProc = rank-1            
            data = comm.recv(None, source=fromProc)
            print(f'El proceso {rank} recibio el mensaje {data} del proceso {fromProc}', flush=True)

            nextProc = 0 if (rank + 1) >= size else rank + 1
            comm.send(message, dest=nextProc)
            print(f'El proceso {rank} envia el mensaje {message} al proceso {nextProc}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Anillo', description='Computacion Avanzada')
    parser.add_argument('-c', help='Cantidad de ciclos del anillo', type=int, required=True)
    args = parser.parse_args()
    n = args.c
    main(n)



