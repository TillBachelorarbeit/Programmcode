#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import math as m
from matplotlib.collections import PolyCollection
import seaborn as sns


# Um Matrix Elemente aendern zu koennnen
class Var:
  def __init__(self, value):
    self.value = value

  def get_conj(self):
    return self.value.conjugate()

  def set_value(self, new):
    self.value = new
  
  def get_value(self):
    return self.value
  


class Hamiltonian:
  # t= Tunnelwahrscheinlichkeiten in t = [t_0, t_eg, t_ee] Reihenfolge,t
  # mit
  # t_0 = g1 in g2, also g in g
  # t_eg = g1 in e2, also g in e
  # t_ee = e1 in e2, also e in e
  # e = Energie liste (Inhalt auch vom Typen Var)

  def __init__(self,  n_topf, e:list ,t:list) -> None:
    self.n_topf = 2   # Anzahl Toepfe
    self.is_periodisch = False 

    self.e = [Var(e[0]), Var(e[1]), Var(e[2]), Var(e[3])]

    self.t_0 = Var(t[0])
    self.t_eg = Var(t[1])
    self.t_ee = Var(t[2])
    
    self.t_0_c = Var(t[0].conjugate())
    self.t_eg_c = Var(t[1].conjugate())
    self.t_ee_c = Var(t[2].conjugate())
    
    self.H = np.array([[self.e[0], 0, self.t_0, self.t_eg], \
                      [0, self.e[1], self.t_eg_c, self.t_ee], \
                      [self.t_0_c, self.t_eg, self.e[2], 0], \
                      [self.t_eg_c, self.t_ee_c, 0, self.e[3]]])

    # erstellt Hamilton mit n Toepfen
    while (self.n_topf != n_topf):
      self.__add_well(e[2* self.n_topf], e[2*self.n_topf +1])
      

  def calculate_ew_and_ez(self):
    self.__fix_values()
    self.ew, ez_temp = np.linalg.eigh(self.H_fix_values)
    self.ez = []
    for i in range(len(ez_temp[0])):
      self.ez.append(np.array(ez_temp[:, i]))
    return self.ew, self.ez

  def calculate_basis(self):
    n = 2* self.n_topf
    self.basis = np.zeros((n,n))
    for i in range(n):
      self.basis[i][i] = 1
    return self.basis

  def periodisch(self):
    self.is_periodisch = True
    l = len(self.H[0])
    
    self.H[0][l-2] = self.t_0
    self.H[0][l-1] = self.t_eg

    self.H[1][l-2] = self.t_eg_c
    self.H[1][l-1] = self.t_ee
    
    self.H[l-2][0] = self.t_0_c
    self.H[l-2][1] = self.t_eg
    
    self.H[l-1][0] = self.t_eg_c
    self.H[l-1][1] = self.t_ee_c   
 
  def change_tunnelwk(self, t:list):
    self.t_0.set_value(t[0])
    self.t_eg.set_value(t[1])
    self.t_ee.set_value(t[2])
    
    self.t_0_c.set_value(t[0].conjugate())
    self.t_eg_c.set_value(t[1].conjugate())
    self.t_ee_c.set_value(t[2].conjugate())

  def change_energy(self, e:list):
    if len(e) != len(self.e):
      return "Error: wrong size of input \n"
    for i in range(len(e)):
      self.e[i].set_value(e[i])


  # psi_start = in welchen Topf startet system. Dabei bedeutet:
  # 0 = Grundzustand 1. Topf, 1 = angeregter Zustand 1. Topf,
  # 2 = Grundzustand 2. Topf, ...
  # t_start, t_end: wann starten/enden
  # grundzustand: True, wenn nur grundzustaende betrachtet werden sollen
  # save_fig: True, wenn plot gespeichert werden soll
  # highes_prob: Welches soll der hoechste Wert in der heatmap sein
  # ylabel: True, wenn y-achse beschriftet sein soll
  # psi_start_state: wenn Teilchen in mehreren Toepfen starten soll, 
  #                   z.B. gauss verteilet
  # y_label_range: Topfnummer, wenn y-achse beschrifftet werden soll                  
  def heatmap(self, psi_start = 0, t_start = 0, t_end = 15,\
              grundzustand = False, save_fig = False, highes_prob = 0,\
              ylabel=False, psi_start_state = [], y_label_range = []):
    
    # Erstellt Daten
    time_points = 5
    t_heatmap = np.linspace(t_start, t_end, (t_end-t_start) * time_points)
    self.calculate_basis()
    psi_0 = self.basis[psi_start]
    if psi_start_state != []:
      psi_0 = psi_start_state
    self.calculate_ew_and_ez()
  
    if grundzustand:
      data_heatmap = np.zeros((self.n_topf, len(t_heatmap)))
      j = 0 
      for i in range(0, 2* self.n_topf, 2):
        data_heatmap[j] = [self.__wkeit(self.basis[i], \
                          self.__calculate_psi_t(psi_0,t)) for t in t_heatmap]        
        j += 1
    else:
      data_heatmap = np.zeros((2 * self.n_topf, len(t_heatmap)))
      for i in range(0, 2* self.n_topf):
        data_heatmap[i] = [self.__wkeit(self.basis[i], \
                          self.__calculate_psi_t(psi_0,t)) for t in t_heatmap]
    
    # Plotten der Daten
    fig, ax = plt.subplots()
    
    if highes_prob != 0:
      heat_map = sns.heatmap(data_heatmap, vmin=0 , vmax=highes_prob,\
                              cbar=False)
    else:
      heat_map = sns.heatmap(data_heatmap, vmin=0, cbar=False)
    
    #mit y - Achsen Beschrifftung
    if ylabel:
      label_name = []
      # Wenn Topfnummern anders als 1,...,n sein sollen
      if y_label_range != []:
        for i in y_label_range:
          label_name.append(f'{i}')
          label_name.append(f'T{i}, e')
      else:
        for i in range(1,self.n_topf+1):
          label_name.append(f'T{i}, g')
          label_name.append(f'T{i}, e') 
      
      
      if grundzustand:
        ax.set_yticks([i-0.5 for i in range(1, self.n_topf + 1,10)])
        ax.set_yticklabels(label_name[::20], rotation=0, fontsize=16)
      else: 
        ax.set_yticks([i-0.5 for i in range(1, 2 * self.n_topf + 1)])
        ax.set_yticklabels(label_name, rotation=0, fontsize=16)
    else:
      # keine y werte
      ax.set_yticks([])
    
    # x achse 
    time_steps = time_points *  (t_start + t_end) // 4
    xticks = [int(i*time_points) for i in t_heatmap[::time_steps]] + \
              [int(t_heatmap[-1] *time_points )]
    xticks_labels = [f'{i:.1f}' for i in t_heatmap[::time_steps]] + \
                    [f'{t_heatmap[-1]:.1f}']
    ax.set_xticks(xticks )
    ax.set_xticklabels(xticks_labels, rotation=0, fontsize=12)
    
    ax.set_xlabel('t', fontsize=16)
    ax.set_ylabel('Topfnummer', fontsize=16)
    plt.tight_layout()
    if save_fig:
      if self.is_periodisch:
        plt.savefig(f'plots/{self.n_topf}_heatmap_periodisch.png', dpi = 300)
      else:
        plt.savefig(f'plots/{self.n_topf}_heatmap.png', dpi = 300)
    else:
      plt.show()
  
  def erwartungswert(self, psi_start = 0, t_start = 0, t_end = 15,\
                      grundzustand = False, save_fig = False):    
    t_list = np.linspace(t_start, t_end, t_end * 5)
    self.calculate_basis()
    psi_0 = self.basis[psi_start]
    self.calculate_ew_and_ez()
    
    label_name = []
    for i in range(1,self.n_topf+1):
      label_name.append(f'T{i}, g')
      label_name.append(f'T{i}, e')
    
    y = []
    
    if grundzustand:
      for t in t_list:
        psi_t = self.__calculate_psi_t(psi_0,t)
        temp = [self.__wkeit(self.basis[i], psi_t) * (i+1) \
                for i in range(0,self.n_topf * 2,2)]
        y.append(sum(temp))
    else:
      for t in t_list:
        psi_t = self.__calculate_psi_t(psi_0,t)
        temp = [self.__wkeit(self.basis[i], psi_t) * (i+1) \
                for i in range(self.n_topf * 2)]
        y.append(sum(temp))
   
    plt.style.use('seaborn')
    
    fig, ax = plt.subplots()
    ax.plot(t_list, y)

    if grundzustand:
      ax.set_yticks([i for i in range(1, 2 * self.n_topf + 1,2)])
      ax.set_yticklabels(label_name[::2])
    else:
      ax.set_yticks([i for i in range(1, 2 * self.n_topf + 1)])
      ax.set_yticklabels(label_name)
    
      
    ax.set_xlabel('t')
    ax.set_ylabel('Zustand')
    plt.tight_layout()
    if save_fig:
      if self.is_periodisch:
        plt.savefig(f'plots/{self.n_topf}_Erwatungswert_periodisch.pdf', \
                      dpi = 300)
      else:
        plt.savefig(f'plots/{self.n_topf}_Erwatungswert.pdf', dpi = 300)
    else:
      plt.show()
    plt.clf()
  
  # paper:
  # "T. Hartmann, F. Keck, H. J. Korsch, and S. Mossmann. 
  # Dynamics of bloch oscillations. New Journal of Physics, 6:2–2, jan 2004."
  
  # paper_single_site: ew und std von der paper oben, wobei:
  # paper_single_site[0] = gamma, paper_single_site[1] = omega_B
  # paper_gauss: ew und std von der paper oben, wobei:
  # paper_gauss[0] = omega_b, paper_gauss[1] = delta, 
  #                  paper_gauss[2] = kappa_0, paper_gauss[3] = d
  
  def erwartungswert_sd(self, psi_start = 0, t_start = 0, t_end = 15, \
                        grundzustand = False, save_fig = False,\
                        ylabel_all_state=False, paper_single_site=[None,None],\
                        paper_gauss=[None,None,None,None], psi_start_state=[],\
                        y_label_range=[]):  
    plt.style.use('seaborn')  
    fig, ax = plt.subplots()
    
    t_list = np.linspace(t_start, t_end, t_end * 5)
    self.calculate_basis()
    psi_0 = self.basis[psi_start]
    if psi_start_state != []:
      psi_0 = psi_start_state
    self.calculate_ew_and_ez()
    
    ev = []   # Erwartungswert
    ev_square = []  # Erwartungswert von nˆ2
    sd = []   # Standardabweichung
    if grundzustand:
      for t in t_list:
        psi_t = self.__calculate_psi_t(psi_0,t)

        if y_label_range != []:
          j = 0
          temp_ev = []
          temp_ev_square = []
          
          for i in y_label_range:
            wkeit = np.real(np.vdot(psi_t[j],psi_t[j]))
            temp_ev.append(wkeit * i)
            temp_ev_square.append(wkeit * i**2)
            j += 2
          
        else:
          temp_ev = [np.real(np.vdot(psi_t[i],psi_t[i])) * (i+1)\
                  for i in range(0,self.n_topf * 2,2)]
          temp_ev_square = [np.real(np.vdot(psi_t[i],psi_t[i])) * (i+1)**2\
                          for i in range(0,self.n_topf * 2,2)] 
    
        ev_sum = round(sum(temp_ev),3)
        ev_sum_square = round(sum(temp_ev_square),3)
        ev.append(ev_sum)
        ev_square.append(ev_sum_square)
        sd.append(m.sqrt( ev_sum_square -  ev_sum**2))
        
    else:
      for t in t_list:
        psi_t = self.__calculate_psi_t(psi_0,t)
        temp_ev = [np.real(np.vdot(psi_t[i],psi_t[i])) * (i+1)\
                  for i in range(self.n_topf * 2)]
        temp_ev_square = [np.real(np.vdot(psi_t[i],psi_t[i])) * (i+1)**2\
                          for i in range(self.n_topf * 2)]
        ev_sum = round(sum(temp_ev),3)
        ev.append(ev_sum)
        sd.append(m.sqrt( round(sum(temp_ev_square),3) -  ev_sum**2))  
        
      
    if paper_gauss != [None,None,None,None]:
      # paper_gauss[0] = omega_b, paper_gauss[1] = delta, 
      # paper_gauss[2] = kappa_0, paper_gauss[3] = d
      h_quer = 1
      u_t = [1 / paper_gauss[0] * m.sin(paper_gauss[0]*t) for t in t_list]
      v_t = [1 / paper_gauss[0] * (1 - m.cos(paper_gauss[0]*t)) for t in t_list]
      temp = paper_gauss[2] * paper_gauss[3]
      ev_paper = [(-paper_gauss[1]/ (2* h_quer)) * \
                  (v_t[i] * m.cos(temp) - u_t[i] * m.sin(temp)) \
                  for i in range(len(t_list))]

      sd_paper = m.sqrt(ev_square[0])
      sd_unten_paper = [ev_paper[i] - sd_paper for i in range(len(ev))]
      sd_oben_paper = [ev_paper[i] + sd_paper for i in range(len(ev))]
        
      ax.plot(t_list, ev_paper, 'r',linewidth=3, \
              label='Erwartungswert nach Formeln')
      ax.plot(t_list, sd_unten_paper, 'r-.', \
              label='Standardabweichung nach Formeln')
      ax.plot(t_list, sd_oben_paper, 'r-.')
  
    
    if paper_single_site != [None,None]:
      sd_paper = [m.sqrt(2) * paper_single_site[0] * \
                  abs( m.sin( paper_single_site[1] * t / 2) ) for t in t_list]  
      sd_unten_paper = [ev[i] - sd_paper[i] for i in range(len(ev))]
      sd_oben_paper = [ev[i] + sd_paper[i] for i in range(len(ev))]
    
      ax.plot(t_list, [0] * len(t_list), "r",linewidth=3, \
              label='Erwartungswert nach Formeln')
      ax.plot(t_list, sd_unten_paper, 'r-.', \
              label='Standardabweichung nach Formeln')
      ax.plot(t_list, sd_oben_paper, 'r-.')
     
    # ploten:

    sd_unten = [ev[i] - sd[i] for i in range(len(ev))]
    sd_oben = [ev[i] + sd[i] for i in range(len(ev))]
    
    ax.plot(t_list, ev, 'b-',label='Erwartungswert')
    ax.plot(t_list, sd_unten, 'b:', label='Standardabweichung')
    ax.plot(t_list, sd_oben, 'b:') 
      
    label_name = []
    if y_label_range != []:
      for i in y_label_range:
        label_name.append(f'{i}')
        label_name.append(f'T{i}, e')
    else:
      for i in range(1,self.n_topf+1):
        label_name.append(f'T{i}, g')
        label_name.append(f'T{i}, e')  
         
    if grundzustand:
      if ylabel_all_state:
        ax.set_yticks(y_label_range)
        ax.set_yticklabels(label_name[::2], fontsize=18)
      else:
        ax.set_yticks(y_label_range[::5])
        ax.set_yticklabels(label_name[::10], fontsize=18)
    else:
      if ylabel_all_state:
        ax.set_yticks([i for i in range(1, 2 * self.n_topf + 1)])
        ax.set_yticklabels(label_name)
      else:
        ax.set_yticks([i for i in range(1, 2 * self.n_topf + 1,5)])
        ax.set_yticklabels(label_name[::5])
    
    ax.tick_params(axis='x', labelsize=14)
      
    ax.set_xlabel('t', fontsize=18)
    ax.set_ylabel('Topfnummer', fontsize=18)
    ax.legend(fontsize=14)
    plt.tight_layout()
    if save_fig:
      if self.is_periodisch:
        plt.savefig(f'plots/{self.n_topf}_Ev_sd_periodisch.pdf', dpi = 300)
      else:
        plt.savefig(f'plots/{self.n_topf}Ev_st.pdf', dpi = 300)
    else:
      plt.show()
    plt.clf()
  
  
  def time_evolution_wflike(self, psi_start = 0, t_start = 0, t_end = 15, \
                            grundzustand = True, save_fig=False):
    t_list = np.linspace(t_start, t_end, t_end)
    self.calculate_basis()
    psi_0 = self.basis[psi_start]
    self.calculate_ew_and_ez()

    label_name = []
    for i in range(1,self.n_topf+1):
      label_name.append(f'T{i}, g')
      label_name.append(f'T{i}, e')
        
    if grundzustand:
      bottom_list = np.array([0.0 for _ in range(len(t_list))])

      
      for i in range(0,self.n_topf * 2, 2):
        y = [self.__wkeit(self.basis[i], self.__calculate_psi_t(psi_0,t)) \
             for t in t_list]
        plt.bar(t_list, y, label=label_name[i], bottom= bottom_list)
        bottom_list += np.array(y)
      
    else:
      bottom_list = np.array([0.0 for _ in range(len(t_list))])
      
      for i in range(self.n_topf * 2):
        y = [self.__wkeit(self.basis[i], self.__calculate_psi_t(psi_0,t)) \
              for t in t_list]
        plt.bar(t_list, y, label=label_name[i], bottom=bottom_list)
        bottom_list += np.array(y)
    
    plt.xlabel('t')
    plt.ylabel('Wahrscheinlichkeit')
    plt.legend(loc='upper right')  
    plt.tight_layout()
    if save_fig:
      if self.is_periodisch:
        plt.savefig(f'plots/{self.n_topf}_wflike_periodisch.pdf', dpi = 300)
      else:
        plt.savefig(f'plots/{self.n_topf}_wflike.pdf', dpi = 300)

    else:
      plt.show()
    plt.clf()
    
  # info: ob informationen wie, energien, tunnelwk's und wo gestartet 
  #       ausgegeben werden sollen
  def time_evolution_waterfall_3d(self, psi_start = 0, t_start = 0, t_end = 9,\
                              grundzustand = True, save_fig=False, info=False):
    
    t_list = np.linspace(t_start, t_end, t_end * 5)
    self.calculate_basis()
    psi_0 = self.basis[psi_start]
    self.calculate_ew_and_ez()

    axes = plt.axes(projection='3d')    #Fuers ploten

    verts = [] 
    z1 = []
    
    if grundzustand:
      label_name = []
      for i in range(1,self.n_topf+1):
        label_name.append(f'T{i}, g')

      for i in range(0,self.n_topf * 2,2):
        z1.append(i)
        y = [self.__wkeit(self.basis[i], self.__calculate_psi_t(psi_0,t)) \
              for t in t_list]
        verts.append(list(zip(t_list, y)))
      
      axes.set_yticks([i for i in range(0, 2 * self.n_topf,2)])
      axes.set_yticklabels(label_name)
        
    else:
      label_name = []
      for i in range(1,self.n_topf+1):
        label_name.append(f'T{i}, g')
        label_name.append(f'T{i}, e')

      for i in range(self.n_topf * 2):
        z1.append(i)
        y = [self.__wkeit(self.basis[i], self.__calculate_psi_t(psi_0,t))
              for t in t_list]
        verts.append(list(zip(t_list, y)))

      axes.set_yticks([i for i in range(2 * self.n_topf)])
      axes.set_yticklabels(label_name)
    
    # polt sachen
    poly = PolyCollection(verts, closed=False, edgecolors = 'blue', lw=1 , \
                          facecolor=(0,0,0,0))
    
    axes.add_collection3d(poly, zs=z1, zdir='y')
    
    axes.set_xlabel('t')
    axes.set_xlim3d(0, t_end)
    axes.set_ylabel('Toepfe')
    axes.set_zlabel('Wahrscheinlichkeit')
    axes.set_zlim3d(0, 1)
    axes.set_title("3D Waterfall plot")

    if save_fig:
      if self.is_periodisch:
        plt.savefig(f'plots/{self.n_topf}_waterfall3d_p.pdf', dpi = 300)
      else:
        plt.savefig(f'plots/{self.n_topf}_waterfall3d.pdf', dpi = 300)
    else:
      plt.show()
    plt.clf()
    
    if info:
      print(f'gestartet im Zustand {label_name[psi_start]}')
      self.__print_informations()
  
  # mean: Zeitlich gemittelte W'keit  
  def time_evolution(self, psi_start = 0, t_start = 0, t_end = 15, \
                      grundzustand = False, save_fig=False, info=False,\
                      mean=False):
    t_list = np.linspace(t_start, t_end, t_end * 5)
    self.calculate_basis()
    if grundzustand:
      psi_start = 2*psi_start
    psi_0 = self.basis[psi_start]
    self.calculate_ew_and_ez()
    
    label_name = []
    for i in range(1,self.n_topf+1):
      label_name.append(f'T{i}, g') 
      label_name.append(f'T{i}, e')

    if mean:
      print('Aufenthaltswahrscheinlichkeit gemittelt:')  
      
    if grundzustand:
      for i in range(0,self.n_topf * 2, 2):
        y = [self.__wkeit(self.basis[i], self.__calculate_psi_t(psi_0,t)) \
              for t in t_list]
        if mean:
          print(f'{label_name[i]}: {sum(y)/len(t_list):.4f}')
        plt.plot(t_list, y, label=label_name[i])
    else:
      for i in range(self.n_topf * 2):
        y = [self.__wkeit(self.basis[i], self.__calculate_psi_t(psi_0,t)) \
              for t in t_list]
        if mean:
          print(f'{label_name[i]}: {sum(y)/len(t_list):.4f}')
        plt.plot(t_list, y, label=label_name[i])
     
    if info:
      print(f'gestartet im Zustand {label_name[psi_start]}') 
      self.__print_informations()
     
    plt.xlabel('t',fontsize=16)
    plt.ylabel('Wahrscheinlichkeit',fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.legend(loc='upper right', fontsize=23)
    plt.tight_layout()
    if save_fig:
      if self.is_periodisch:
        plt.savefig(f'plots/{self.n_topf}_time_evolution_p.pdf', dpi = 300)
      else:
        plt.savefig(f'plots/{self.n_topf}_time_evolution.pdf', dpi = 300)
    else:
      plt.show()
    plt.clf()

  
  # bar plot zum Zeitpunkt t
  def bar_plot_wkeit_time(self, t, psi_start = 0):
    assert(psi_start < 2* self.n_topf)
    self.calculate_basis()
    psi_0 = self.basis[psi_start]
    self.ew, self.ez = self.calculate_ew_and_ez()
    self.bar_plot_wkeit(psi_t= self.__calculate_psi_t(psi_0, t))

  # num ist die nummer des angeregtem Zustandes des gesamtsystems
  # Bsp.: num = 1: 1. Angeregter Zustand des gesamtsystems
  def bar_plot_wkeit(self, num = 0 , all = False, psi_t = [], save_fig=False, \
                      info=False):
    plt.style.use('seaborn')
    
    ew, ez = self.calculate_ew_and_ez()
    self.calculate_basis()

    # psi_t ist da um wkeit plot zum zeitpunkt t im System zu bestimmen
    if psi_t != []:
      ez = [psi_t]  # Das als ez zu bezeichnen ist falsch, ist aber einfacher
      
     
      
    if num >= len(ez):
      print(f'num have to be smaller then {len(ez)}')
      return -1
    
    # fuer x label beim Plot
    x = []
    for i in range(1,int(ew.size//2)+1):
      x.append(f'T{i}, g')
      x.append(f'T{i}, e')
    
    if info:
      self.__print_informations()
    
    if all:
      for j in range(2 * self.n_topf):
        y = [self.__wkeit(self.basis[i], ez[j]) \
              for i in range(2 * self.n_topf)]

        print(f'Mit {j+1}. Eigenzustand \n')
        plt.bar(x,y)
        plt.xlabel('Topf und Zustand [Topf, Zustand]')
        plt.ylabel('Wahrscheinlichkeit')
        plt.tight_layout()
        plt.show()
    else:
      y = [self.__wkeit(self.basis[i], ez[num]) \
            for i in range(2 * self.n_topf)]
      
      # wenn entartung vorliegt:
      # y1 = [self.__wkeit(self.basis[i], ez[0]) \
      #       for i in range(2 * self.n_topf)]
      # y2 = [self.__wkeit(self.basis[i], ez[1]) \
      #       for i in range(2 * self.n_topf)]
      # y = [1/m.sqrt(2) * (y1[i] + y2[i]) for i in range(len(y1))]
      
      if psi_t == []:
        print(f'Mit {num + 1}. Eigenzustand \n')
      plt.bar(x,y)
      plt.xlabel('Topf und Zustand [Topf, Zustand]', fontsize=18)
      plt.ylabel('Wahrscheinlichkeit', fontsize=18)
      plt.tight_layout()
      plt.tick_params(axis='x', labelsize=18)
      plt.tick_params(axis='y', labelsize=18)
      
      if save_fig:
        if self.is_periodisch:
          plt.savefig(f'plots/{self.n_topf}_bar_plot_p.pdf', dpi = 300)
        else:
          plt.savefig(f'plots/{self.n_topf}_bar_plot.pdf', dpi = 300)
      else:
        plt.show()
      plt.clf()
    plt.style.use('default')
    
  def __print_informations(self):
    tunnelwk = self.get_tunnelwk()
    print('\n')
    print(f'Energien = {self.get_energy()}')
    print(f'Tunnelwkeit von Grundzustand in Grundzustand = {tunnelwk[0]}')
    print(f'Tunnelwkeit von angeregten in Grundzustand = {tunnelwk[1]} \
          (nur fuer unterschiedliche Toepfe)')
    print(f'Tunnelwkeit von angeregten in angeregten Zustand = {tunnelwk[2]}')  
    print('\n')
  
  # Fixiert die Werte von den Hamilton (davor waren die Werte des Hamiltons
  # vom Typ Var. Allerdings werden zur weiteren Berechnung integer oder float
  # Zahlen bentötigt)  
  def __fix_values(self):
    H_temp = [ [] for i in range(len(self.H))]
                                      
    for i in range(len(self.H)):
      for j in range(len(self.H[0])):
        if type(self.H[i][j]) == type(1) or type(self.H[i][j]) == type(1.0):
          H_temp[i].append(self.H[i][j])
        else:
          H_temp[i].append(self.H[i][j].get_value())
    self.H_fix_values = np.array(H_temp)

  def __add_well(self, e_i0, e_i1):
    self.n_topf += 1
    self.e.append(Var(e_i0))
    self.e.append(Var(e_i1))
    H_new = []

    for i in range(len(self.H)):
      if i < len(self.H) - 2:
        temp_row = list(self.H[i])
        temp_row.append(0)
        temp_row.append(0)
        H_new.append(temp_row)
      elif i == (len(self.H) -2):
        temp_row = list(self.H[i])
        temp_row.append(self.t_0)
        temp_row.append(self.t_eg)
        H_new.append(temp_row)
      else:
        temp_row = list(self.H[i])
        temp_row.append(self.t_eg_c)
        temp_row.append(self.t_ee)
        H_new.append(temp_row)

    temp_row = [0] * (len(self.H) - 2)
    temp_row.append(self.t_0_c)
    temp_row.append(self.t_eg)
    temp_row.append(self.e[2 * (self.n_topf - 1)]) 
    temp_row.append(0)
    H_new.append(temp_row)

    temp_row = [0] * (len(self.H) - 2)
    temp_row.append(self.t_eg_c)
    temp_row.append(self.t_ee_c)
    temp_row.append(0)
    temp_row.append(self.e[2 * self.n_topf - 1])
    H_new.append(temp_row)
    self.H = np.array(H_new)

  def __calculate_psi_t(self, psi_0, t):
    h = 1  # hbar = 1 gesetzt
    self.psi = np.array([0+0j for _ in range(2* self.n_topf)])
    for n in range(len(self.ew)):
      temp1 = np.exp(-1j * self.ew[n] * t / h) * np.vdot(self.ez[n],psi_0)
      self.psi +=  temp1 * self.ez[n]
    return self.psi

  def __wkeit(self, psi, chi):
    assert(len(psi) == len(chi))
    ska = np.vdot(psi,chi) 
    return  np.real(np.vdot(ska,ska))

  def print_matrix(self):
    for i in range(len(self.H)):
      for j in range(len(self.H[0])):
        if type(self.H[i][j]) == type(1) or type(self.H[i][j]) == type(1.0):
          print(f'{self.H[i][j]:<4}', end=' ')
        else:
          print(f'{self.H[i][j].get_value():<4}',end=" ") 
      print('')
    print('\n')

  def print_ez(self):
    for i in self.calculate_ew_and_ez()[1]:
      print(i)
     
  def print_ew(self):
    for i in self.calculate_ew_and_ez()[0]:
      print(f'{i:.2f}')

  def get_matrix_with_values(self):
    self.__fix_values()
    return self.H_fix_values

  def get_energy(self):
    return [self.e[i].get_value() for i in range(len(self.e))]

  def get_tunnelwk(self):
    return [self.t_0.get_value(), self.t_eg.get_value(), self.t_ee.get_value()]

  def get_anzahl_topf(self):
    return self.n_topf
  
  def get_ez(self):
    return self.calculate_ew_and_ez()[1]
     
  def get_ew(self):
    return self.calculate_ew_and_ez()[0]
