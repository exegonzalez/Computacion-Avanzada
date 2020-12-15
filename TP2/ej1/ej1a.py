#!/usr/bin/env python3

from mpi4py import MPI

comm = MPI.COMM_WORLD
node = comm.Get_rank()

print(f'Hello world fromNode {node}')