#!/usr/bin/env python3

import import_hamiltonian
import hamiltonian
import math as m
import help_functions as pf  # importiere benoetigte Funktionen
import numpy as np
import sys

def usage():
  print(f"1. Fall mit 81 Toepfen {sys.argv[0]} 1")
  print(f"2. Fall mit 61 Toepfen {sys.argv[0]} 2")
 

if __name__ == '__main__':

  if len(sys.argv) != 2:
    sys.stderr.write(f'{sys.argv[0]} needs an argument \n')
    usage()
    exit(0)
    
  if sys.argv[1] == "1":
    # Parameter  
    n = 81
    
    F = 0.005        # Kraft
    d = 2*m.pi       # Peridoizitaet vom Potential
    Tw = 0.2485      # Tunnelwahrscheinlichkeitsamplitude
    t_end = int(2  * ( 2 * m.pi / (d * F)) + 1)  # t fuer 2 Perioden
    kappa = 0       # Kappa_0, anfangs Impuls
    
    # Interessante Werte
    T =  2 * m.pi / (d * F)   # Bloch Periode
    omega = 2 * m.pi / T      # Bloch Frequenz
    gamma = 2*Tw / (d*F)


    # Berechne Energien vom Hamiltonian
    n_range = [i for i in range(20, -(60+ 1),-1)]

    psi_0_g = [2 * pf.gaus(i, beta=0.01, kappa=kappa, d=d) for i in n_range]
    psi_0_e = [0 for i in n_range]
    psi_0_t = pf.zipping(psi_0_g,psi_0_e)
    psi_0 = 1/ m.sqrt(pf.norm(psi_0_t,psi_0_t)) * np.array(psi_0_t)

    energie_g = [d * F * i for i in n_range]
    energie_e = [0 for _ in n_range]
    energie = pf.zipping(energie_g,energie_e)


    # Erstellt Hamiltonian
    h = hamiltonian.Hamiltonian(n, energie, [-Tw ,0 ,0])
    h.periodisch()
    
    # Heatmap   
    h.heatmap(t_end=t_end, grundzustand=True, ylabel=True, \
              psi_start_state=psi_0, y_label_range=n_range) 

    # Erwartungswert mit Standardabweichung 
    # paper_gauss[0] = omega_b, paper_gauss[1] = delta, 
    # paper_gauss[2] = kappa_0, paper_gauss[3] = d
    h.erwartungswert_sd(t_end=t_end, grundzustand=True, psi_start_state=psi_0,\
                        ylabel_all_state=False, \
                        paper_gauss=[omega,Delta, kappa, d],\
                        y_label_range=n_range)

  elif sys.argv[1] == "2":
    # Parameter  
    n = 61
    
    F = 0.005        # Kraft
    d = 2*m.pi       # Peridoizitaet vom Potential
    Tw = 0.2485      # Tunnelwahrscheinlichkeitsamplitude
    t_end = int(2  * ( 2 * m.pi / (d * F)) + 1)  # t fuer 2 Perioden
    kappa = 0       # Kappa_0, anfangs Impuls
    
    # Interessante Werte
    T =  2 * m.pi / (d * F)   # Bloch Periode
    omega = 2 * m.pi / T      # Bloch Frequenz
    gamma = 2*Tw / (d*F)


    # Berechne Energien vom Hamiltonian
    n_range = [i for i in range(20, -(40+ 1),-1)]

    psi_0_g = [2 * pf.gaus(i, beta=0.3, kappa=kappa, d=d) for i in n_range]
    psi_0_e = [0 for i in n_range]
    psi_0_t = pf.zipping(psi_0_g,psi_0_e)
    psi_0 = 1/ m.sqrt(pf.norm(psi_0_t,psi_0_t)) * np.array(psi_0_t)

    energie_g = [d * F * i for i in n_range]
    energie_e = [0 for _ in n_range]
    energie = pf.zipping(energie_g,energie_e)


    # Erstellt Hamiltonian
    h = hamiltonian.Hamiltonian(n, energie, [-T ,0 ,0])
    h.periodisch()
    
    # Heatmap   
    h.heatmap(t_end=t_end, grundzustand=True, ylabel=True, \
              psi_start_state=psi_0, y_label_range=n_range) 

    # Erwartungswert mit Standardabweichung 
    # paper_gauss[0] = omega_b, paper_gauss[1] = delta, 
    # paper_gauss[2] = kappa_0, paper_gauss[3] = d
    h.erwartungswert_sd(t_end=t_end, grundzustand=True, psi_start_state=psi_0,\
                        ylabel_all_state=False, \
                        paper_gauss=[omega,Delta, kappa, d],\
                        y_label_range=n_range)
