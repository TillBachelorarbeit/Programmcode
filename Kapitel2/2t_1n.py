#!/usr/bin/env python3
# Oszillation der Aufenhthaltsw'keit fur linken und rechten Topf

import matplotlib.pyplot as plt
import numpy as np

# hbar = 1 gesetzt, da sonst zu viele Oszillationen
T = 1  # Tunnelwkeitamplitude

  
def wkeit_l_unterschiedlich(el, er, T, t):
  gamma = 4 * T**2 / (el - er)**2
  frequenz = 1/2 * np.sqrt((el - er)**2 + 4* T**2 )  
  t1 = gamma**2 / (4*(1 + np.sqrt(1 + gamma) + gamma)**2 )
  t2 = gamma**2 / (4*(1 - np.sqrt(1 + gamma) + gamma)**2 )
  amplitude = (gamma / (2 * (gamma + 1 )))
  t3 = amplitude * np.cos(t * frequenz)  
  return t1 + t2 + t3

def wkeit_r_unterschiedlich(el, er, T, t):
  gamma = 4 * T**2 / (el - er)**2
  frequenz = 1/2 * np.sqrt((el - er)**2 + 4* T**2 )  
  t1 = (gamma * (1 + np.sqrt(1 + gamma))**2 ) / \
        (4*(1 + np.sqrt(1 + gamma) + gamma)**2 )
  t2 = (gamma * (1 - np.sqrt(1 + gamma))**2 ) / \
        (4*(1 - np.sqrt(1 + gamma) + gamma)**2 )
  amplitude = (gamma / (2 * (gamma + 1 )))
  t3 = amplitude * np.cos(t * frequenz)  
  return t1 + t2 - t3

n = 4 * np.pi  # bis wohin
x = np.linspace(0, n, 100)

fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1)

ax1.plot(x, [wkeit_l_unterschiedlich(3.001, 3, T, i) for i in x], 'b' ,\
        label="erster Topf")
ax1.plot(x, [wkeit_r_unterschiedlich(3.001, 3, T,i) for i in x], 'g' ,\
        label="zweiter Topf")
ax1.set_ylabel('Wahrscheinlichkeit')
ax1.legend(loc='best')

ax2.plot(x, [wkeit_l_unterschiedlich(5, 1, T, i) for i in x], 'b' ,\
        label="erster Topf")
ax2.plot(x, [wkeit_r_unterschiedlich(5, 1, T,i) for i in x], 'g' ,\
        label="zweiter Topf")
ax2.set_xlabel('t')
ax2.set_ylabel('Wahrscheinlichkeit')


# plt.plot(x, [wkeit_l_unterschiedlich(5, 3, T, i) for i in x], 'b' , \
#         label="erster Topf")
# plt.plot(x, [wkeit_r_unterschiedlich(5, 3, T,i) for i in x], 'g' ,\
#           label="zweiter Topf")
# plt.xlabel('t')
# plt.ylabel('Wahrscheinlichkeit')

plt.savefig('oszillation.pdf', dpi=300)
plt.show()
  