#! /usr/bin/env python3

# ===================================================================================
# Name: prl_life.py
# Description: Sequence implementation of Conways Life.
# Author: Milan Munzar (xmunza00@stud.fit.vutbr.cz)

import sys
import numpy
import queue as Queue
import threading
import argparse

from gui_mainWindow import MainWindow
from mpi4py import MPI


parser = argparse.ArgumentParser(prog='Game of Life Paralelo', description='Computacion Avanzada')
parser.add_argument('-m', help='Tamaño de la matriz', type=int, required=True)
parser.add_argument('-c', help='Numero de ciclos', type=int, required=True)
args = parser.parse_args()

MATRIX_SIZE = args.m
NO_EPOCH = args.c

# ===================================================================================
# parallel life
def parallelLife(m_act, queue):


    no_rows = len(m_act)
    ms = [m_act, numpy.zeros((no_rows, MATRIX_SIZE))]
    m_activ_w = numpy.ones((no_rows, MATRIX_SIZE), dtype = bool)

    epoch = 0
    while epoch < NO_EPOCH:

        # receives msgs
        halo_up = numpy.zeros(MATRIX_SIZE, dtype = int)
        halo_down = numpy.zeros(MATRIX_SIZE, dtype = int)

        if rank != 0:
            req2 = comm.Irecv(halo_up, source = rank - 1, tag = 2)
        if rank != size - 1:
            req1 = comm.Irecv(halo_down, source = rank + 1, tag = 1)

        # sends halo region
        index = epoch % 2
        m_r = ms[index]
        m_w = ms[not(index)]
        m_activ_r = m_activ_w 
        m_activ_w = numpy.zeros((no_rows, MATRIX_SIZE), dtype = bool) 

        if rank != 0:
            for i in range(1, MATRIX_SIZE - 1):
                halo_up[i] = m_r[0][i - 1] +  m_r[0][i] +  m_r[0][i + 1]  

            comm.Isend(halo_up, dest = rank - 1, tag = 1)

        if rank != size - 1: 
            for i in range(1, MATRIX_SIZE - 1):
                halo_down[i] = m_r[no_rows - 1][i - 1] +  m_r[no_rows - 1][i] + \
                   m_r[no_rows - 1][i + 1]

            comm.Isend(halo_down, dest = rank + 1, tag = 2)

        # computes inner matrix
        for ro in range(1 , no_rows - 1):
            for co in range(1, MATRIX_SIZE - 1): 

                if m_activ_r[ro][co] == False:
                    continue

                m_w[ro][co] = m_r[ro - 1][co - 1] + m_r[ro - 1][co] + m_r[ro - 1][co + 1] + \
                    m_r[ro][co - 1] + m_r[ro][co + 1] + \
                    m_r[ro + 1][co - 1] + m_r[ro + 1][co] + m_r[ro + 1][co + 1]

        # update borders 
        if rank != 0:
            req2.Wait()
            for co in range(1, MATRIX_SIZE - 1):
                
                if m_activ_r[0][co] == False:
                    continue

                m_w[0][co] = halo_up[co] + \
                    m_r[0][co - 1]  + m_r[0][co + 1] + \
                    m_r[1][co - 1] + m_r[1][co] + m_r[1][co + 1] 

        if rank != size - 1:
            req1.Wait()
            for co in range(1, MATRIX_SIZE - 1):

                if m_activ_r[no_rows - 1][co] == False:
                    continue

                m_w[no_rows - 1][co] = halo_down[co] + \
                    m_r[no_rows - 1][co - 1]  + m_r[no_rows - 1][co + 1] + \
                    m_r[no_rows - 2][co - 1] + m_r[no_rows - 2][co] + m_r[no_rows - 2][co + 1] 

        # compute cells
        for ro in range(0, no_rows):
            for co in range(1, MATRIX_SIZE - 1):

                if m_r[ro][co] == 255:
                    if m_w[ro][co] == 510 or m_w[ro][co] == 765:
                        m_w[ro][co] = 255 
                    else:
                        m_w[ro][co] = 0
                        for i in range(0 if ro == 0 else - 1, 1 if ro == no_rows - 1 else 2):
                            for j in range(-1, 2):
                                m_activ_w[ro + i][co + j] = True 
                else:
                    if m_w[ro][co] == 765:
                        m_w[ro][co] = 255 
                        for i in range(0 if ro == 0 else - 1, 1 if ro == no_rows - 1 else 2):
                            for j in range(-1, 2):
                                m_activ_w[ro + i][co + j] = True 
                    else:
                        m_w[ro][co] = 0

        # update GUI
        mt_res = comm.gather(m_w, root = 0)    
        if rank == 0:
            mt_res = numpy.vstack(mt_res)
            queue.put(mt_res)

        epoch = epoch + 1 
    #! parallel life

    return m_w





# ===================================================================================
# Main
if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()  
    q = Queue.Queue()

    # master
    if rank == 0:
        
        # create initial matrix
        mt_initial = numpy.zeros((MATRIX_SIZE, MATRIX_SIZE))
        for ro in range(1, MATRIX_SIZE - 1):
            for co in range(1, MATRIX_SIZE - 1):
                rand = numpy.random.random_integers(1, 10)
                if rand != 10:
                    mt_initial[ro][co] = 0 
                else:
                    mt_initial[ro][co] = 255 

        # gui 
        q.put(mt_initial)
        thread = threading.Thread(target = MainWindow, args = (q,))
        thread.setDaemon(True)
        thread.start()

        # create subarrays 
        mt_div = []
        for i in range(0, len(mt_initial), int(MATRIX_SIZE/size)): 
            mt_div.append(mt_initial[i:i + int(MATRIX_SIZE/size)])

    # slaves
    else:
        mt_div = None

    mt_div = comm.scatter(mt_div, root = 0)
    m_w = parallelLife(mt_div, q) 

    sys.exit(0)
    
# EOF
# ===================================================================================
