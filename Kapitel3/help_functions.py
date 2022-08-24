#!/usr/bin/env python3
import math as m
import cmath as cm
import numpy as np
import matplotlib.pyplot as plt

# Funktionen
def gaus(n, beta, kappa, d):
    g = (2 * beta / m.pi)**(1/4)
    return g * cm.exp(-beta * n**2 + 1j * n * kappa * d)

def norm(psi, chi):
    assert(len(psi) == len(chi))
    ska = np.vdot(psi,chi) 
    return  m.sqrt( np.real(np.vdot(ska,ska)))

def zipping(l1,l2):
    assert(len(l1) == len(l2))
    l = [0] * len(2*l1) 
    for i in range(0,2 * len(l1) -1 ,2):
        l[i] = l1[int(i/2)]
        l[i+1] = l2[int(i/2)]
    return l