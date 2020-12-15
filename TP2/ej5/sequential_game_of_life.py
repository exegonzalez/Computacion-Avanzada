#! /usr/bin/env python3

# ===================================================================================
# Name: seq_life.py
# Description: Sequence implementation of Conways Life.
# Author: Milan Munzar (xmunza00@stud.fit.vutbr.cz)

import sys
import numpy
import timeit
import queue as Queue
import threading
import argparse

from gui_mainWindow import MainWindow

parser = argparse.ArgumentParser(prog='Game of Life Secuencial', description='Computacion Avanzada')
parser.add_argument('-m', help='Tamaño de la matriz', type=int, required=True)
parser.add_argument('-c', help='Numero de ciclos', type=int, required=True)
args = parser.parse_args()

MATRIX_SIZE = args.m
NO_EPOCH = args.c

# ===================================================================================
# sequential life
def sequentialLife(m_act, queue):

    ms = [m_act, numpy.zeros((MATRIX_SIZE, MATRIX_SIZE))]
    m_activ_w = numpy.ones((MATRIX_SIZE, MATRIX_SIZE), dtype = bool)
    epoch = 0

    while epoch < NO_EPOCH:

        #print epoch

        # living cells
        index = epoch % 2
        m_r = ms[index]
        m_w = ms[not(index)]
        # active cells
        m_activ_r = m_activ_w 
        m_activ_w = numpy.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype = bool) 

        # current epoch
        for ro in range(1, MATRIX_SIZE - 1):
            for co in range(1, MATRIX_SIZE - 1):

                if m_activ_r[ro][co] == False:
                    continue

                neighborhood = m_r[ro - 1][co - 1] + m_r[ro - 1][co] + m_r[ro - 1][co + 1] + \
                    m_r[ro][co - 1] + m_r[ro][co + 1] + \
                    m_r[ro + 1][co - 1] + m_r[ro + 1][co] + m_r[ro + 1][co + 1]

                if m_r[ro][co] == 255:
                    if neighborhood == 510 or neighborhood == 765:
                        m_w[ro][co] = 255 
                    else:
                        m_w[ro][co] = 0
                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                m_activ_w[ro + i][co + j] = True 
                else:
                    if neighborhood == 765:
                        m_w[ro][co] = 255 
                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                m_activ_w[ro + i][co + j] = True 
                    else:
                        m_w[ro][co] = 0

        epoch = epoch + 1
        queue.put(m_w)
        #! current epoch
    #! life

    return 

# ===================================================================================
# Main
def main():

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
    queue = Queue.Queue()
    queue.put(mt_initial)
    thread = threading.Thread(target = MainWindow, args = (queue,))
    thread.setDaemon(True)
    thread.start()

    sequentialLife(mt_initial, queue)

    sys.exit(0)

if __name__ == "__main__":
    main()

# EOF
# ===================================================================================
