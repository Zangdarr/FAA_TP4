# -*- coding: utf-8 -*-
"""
Created on Mon Fev 02 10:32:36 2015

@author: alexandre
"""

import numpy as numpy
import matplotlib.pyplot as pyplot


def load_data():
        global N, matrix, f, h, y, x
        f = numpy.loadtxt('x.txt')
        h = numpy.loadtxt('y.txt')
        matrix = numpy.concatenate((f, h), axis=0)
        N = len(matrix)
        x = numpy.vstack((matrix[:,0], numpy.ones(N)))
        y = matrix[:,1]



def pas_batch(val):
        A = 1.0
        B = 10000
        C = 1000
        return ((A/(C + (B * val))))


def f_theta(theta):
        return numpy.dot(theta.T, x)

def print_graphs():
        pyplot.figure(1)
        pyplot.plot(f[:,0],f[:,1], '.', label="femme")
        pyplot.plot(h[:,0],h[:,1], '.', label="homme")
        pyplot.plot(matrix[:,0], f_theta(batch_gradient_descent()), label="theta")
        pyplot.legend()

        pyplot.figure(2)
        xones = numpy.vstack((matrix[:,0],numpy.ones(len(matrix))))
        pyplot.plot(matrix[:,0], sigmoid(numpy.dot(numpy.array(batch_gradient_descent()).T,xones)),".", label="theta")
        pyplot.legend()

        pyplot.figure(3)
        xones = numpy.vstack((matrix[:,1],numpy.ones(len(matrix))))
        pyplot.plot(matrix[:,1], sigmoid(numpy.dot(numpy.array(batch_gradient_descent()).T,xones)),".", label="theta")
        pyplot.legend()
        
        pyplot.show()


def j_theta(theta):
        tmp = (y - numpy.dot(x.T, theta))
        return ((1.0/N) * numpy.dot(tmp.T, tmp))

def batch_gradient_descent():
        theta = [1, 1]
        previous = j_theta(theta)
        current = previous + 1
        i = 1
        while (abs(previous - current) > 10e-6):
                previous = current
                theta = theta + (pas_batch(i) * (1.0/N) * numpy.dot(x, (y - numpy.dot(x.T, theta))))
                i+=1
                current = j_theta(theta)
        print "Batch theta = ", theta
        return theta

def sigmoid(param):
        a = 1
        b = param.mean(0)

        print("A = " + str(a))
        print("\nB = " + str(b))

        b = b *(-1)
        
        print("\n\nA = " + str(a))
        print("\nB = " + str(b))
        
        sig = 1 / (1    + numpy.exp((numpy.dot(a,param) + b)))
        
        return sig

def main():
        load_data()
        print_graphs()


if __name__ == '__main__':
        main()
