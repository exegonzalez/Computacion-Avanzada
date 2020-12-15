#! /usr/bin/env python3

# ===================================================================================
# Name: gui_mainWindow.py
# Description: Implementation of GUI for Conway's life
# Author: Milan Munzar (xmunza00@stud.fit.vutbr.cz)

import sys
import numpy
import queue as Queue

from PIL import ImageTk, Image as PILImage
from tkinter import *

# ===================================================================================
# MaiWindow
class MainWindow():

    def __init__(self, queue):
        """ Initialize UI """
        self.root = Tk() 
        self.root.title("Conway's life")
        self.root.resizable(False, False)
        self.root.minsize(100, 100)

        self.Frame = Frame(self.root) 
        self.Frame.pack()

        self.label = Label(self.Frame)
        self.label.pack()

        self.queue = queue

        self.refresh()
        self.root.mainloop()

    @staticmethod
    def matrixToImage(matrix):
        image = PILImage.fromarray(matrix)
        image = ImageTk.PhotoImage(image)
        return image


    def refresh(self):
        """ Refreshes label """
        if not self.queue.empty():
            self.state = self.queue.get()
            self.image = PILImage.fromarray(self.state)
            self.image = ImageTk.PhotoImage(self.image)
            self.label.configure(image = self.image)
        self.root.after(20, self.refresh)
        return

    def quit(self):
        """ Quits """
        self.root.destroy()
        return        


# ===================================================================================
# Main
if __name__ == "__main__":

    size = 500 
       
    # create initial matrix
    mt_initial = numpy.zeros((size, size))
    for ro in xrange(1, size - 1):
        for co in xrange(1, size - 1):
            rand = numpy.random.random_integers(1, 10)
            if rand != 10:
                mt_initial[ro][co] = 0 
            else:
                mt_initial[ro][co] = 255 

    queue = Queue.Queue()
    queue.put(mt_initial)
    MainWindow(queue)

    sys.exit(0)

# EOF
# ===================================================================================
