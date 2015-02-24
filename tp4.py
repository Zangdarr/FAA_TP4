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

'''
AVANT LES MODIF AVEC ROMAIN
if __name__ == '__main__':
        #Récupération du fichier contenant les tailles des femmes en première colonne et les poids de celles-ci en seconde.
        femme = np.loadtxt("x.txt")
        #Récupération du fichier contenant les tailles des hommes en première colonne et les poids de ceux-ci en seconde.
        homme = np.loadtxt("y.txt")

        #On définit que l'état "Être une femme" est associé à 0
        f_ones = np.zeros(len(femme[:,0]))
        #On définit que l'état "Être un homme" est associé à 1
        h_ones = np.ones(len(homme[:,0]))

        #On fusionne les colonnes respectives des matrices homme et femme
        fusion = np.concatenate((femme,homme), axis=0)

        #On fusionne les colonnes respectives des matrices de 0 et 1
        ones_concat = np.concatenate((f_ones,h_ones), axis=0)

        #Taille de l'ensembre homme/femme
        n = len(fusion)
        
        print("Matrice Fusion : " + str(np.shape(fusion)))
        
        x = fusion[:,0]
        xones = np.vstack((fusion[:,0],np.ones(len(fusion))))     
        y = fusion[:,1]
        yones = np.vstack((np.ones(len(fusion)),fusion[:,1]))   
        
        #batch = descente_gradient(x, y, n)
        batch = descente_gradient(y,xones, n)
        batch.calculer()

        print("Batch: {0}".format(batch.theta))

        
        pl.figure(1)
        pl.plot(fusion[:,0], '*')
        pl.grid(True)
        pl.title("Ensemble des tailles/poids homme/femme")
        
        pl.figure(2)
        pl.axis([140,200,-0.1,1.1])
        pl.plot(homme[:,0],h_ones , '^',label="Homme")
        pl.plot(femme[:,0],f_ones , '*',label="femme")   
        pl.legend(loc='upper left')     
        pl.grid(True)
        pl.title("Representation des tailles homme/femme")
        
        pl.figure(3)
        pl.axis([35,100,-0.1,1.1])
        pl.plot(homme[:,1],h_ones , '^',label="Homme")
        pl.plot(femme[:,1],f_ones , '*',label="femme")   
        pl.legend(loc='upper left')  
        pl.grid(True)
        pl.title("Representation des poids homme/femme")
        pl.show()       
'''
'''
        pl.figure(4)
        pl.plot(bigmama[0,:], bigmama[1,:], '*')
        pl.plot(bigmama[0,:], f_theta(batch.tab_points[-        1], bigmama), '.', label = "Batch")
        print(f_theta(batch.tab_points[-1], bigmama))
'''

        
###### THETA #######


#multiplication de xfinal et de sa transposé
#mul_x_xT = np.dot(x,x.T)
#print mul_x_xT

#inversion de la matrice mul_x_xT
 #inv_mul_x_xT = np.linalg.inv(mul_x_xT)
#print inv_mul_x_xT

#multiplication des matrices x et y
#mul_x_y = np.dot(x,y)
#print mul_x_y

#multiplication de l'inversion et mul_x_y
#mul_inv_xy = np.dot(inv_mul_x_xT, mul_x_y)


#on a trouvé theta
#theta = mul_inv_xy
