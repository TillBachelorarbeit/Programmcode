#!/usr/bin/env python3

import import_hamiltonian
import hamiltonian

H2 = hamiltonian.Hamiltonian(2, [5,10,5,10], [1, 1, 1])
H2.bar_plot_wkeit(save_fig=False)
H2.time_evolution(t_end=10)
H2.time_evolution(t_end=10000, mean=True)

H2.change_energy([5,10,10,15])
H2.change_tunnelwk([1, 1, 0])
H2.bar_plot_wkeit(save_fig=False, num=1)

H2.change_tunnelwk([0, 1, 1])
H2.bar_plot_wkeit(save_fig=False, num=1)
H2.time_evolution(t_end=10, psi_start=1)
H2.time_evolution(t_end=100000, mean=True, psi_start=1)
