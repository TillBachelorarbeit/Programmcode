#!/usr/bin/env python3

import import_hamiltonian
import hamiltonian
import math as m
import help_functions as pf  # importiere benoetigte Funktionen

# Parameter  
n = 81
psi_start = n-2+1  # +1 , da n ungerade 

F = 0.005        # Kraft
d = 2*m.pi       # Peridoizitaet vom Potential
Tw = 0.2485      # Tunnelwahrscheinlichkeitsamplitude
t_end = int(2  * ( 2 * m.pi / (d * F)) + 1)  # t fuer 2 Perioden

# Interessante Werte
T =  2 * m.pi / (d * F)   # Bloch Periode
omega = 2 * m.pi / T      # Bloch Frequenz
gamma = 2*Tw / (d*F)

############################

# Berechne Energien vom Hamiltion
n_range = [i for i in range(n//2, - (n//2 + 1), -1)]
l_g = [d * F * i for i in n_range]
l_e = [0 for _ in n_range]
l = pf.zipping(l_g,l_e)

# Erstellt Hamiltion
h = hamiltonian.Hamiltonian(n, l, [-Tw ,0 ,0])
h.periodisch()
  
# Heatmap   
h.heatmap(t_end=t_end ,psi_start=psi_start, grundzustand=True, \
           highes_prob=0.3, ylabel=True, y_label_range=n_range) 

# Erwartungswert mit Standardabweichung 
h.erwartungswert_sd(t_end=t_end ,psi_start=psi_start, grundzustand=True,\
                    ylabel_all_state=False, paper_single_site=[gamma, omega],\
                    y_label_range=n_range)
