#!/usr/bin/env python3

import import_hamiltonian
import hamiltonian

H3 = hamiltonian.Hamiltonian(3, [5,10,5,10,5,10], [1, 1, 1])
H3.bar_plot_wkeit(save_fig=False)

H3.periodisch()

# Da hier Entartung vorliegt, muss in der hamilton.Hamilton Klasse die
# Funktion  bar_plot_wkeit modifiziert werden und zwar muss der 
# auskommentierte Code-Teil berücksichtigt werden.
H3.bar_plot_wkeit(save_fig=False)
