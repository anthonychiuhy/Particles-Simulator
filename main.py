# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:38:00 2020

@author: Anthony
"""

from objects import Universe, Box, Ball

import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
np.random.seed(0)

box = Box(10, 10, 10)

#x_1, v_1 = np.array([2,5,5]), np.array([3,-4,0])
#x_2, v_2 = np.array([5,5,5]), np.array([2,1,0])
#x_3, v_3 = np.array([8,5,5]), np.array([-3,2,0])

#b1 = Ball(x_1, v_1, r=1, m=1)
#b2 = Ball(x_2, v_2, r=1, m=1)
#b3 = Ball(x_3, v_3, r=1, m=1)

#balls = [b1, b2, b3]

n = 10
xrand = [np.concatenate((np.random.rand(2)*8+1, 5*np.ones(1))) for i in range(n)]
vrand = [np.concatenate((np.random.rand(2)*8-4, np.zeros(1))) for i in range(n)]

balls = [Ball(x=xrand[i], v=vrand[i], m=1, r=0.4) for i in range(n)]

g = np.array([0,-2,0])
universe = Universe(box, balls, g)


plt.subplots(1,1)
for i in range(200):
    plt.pause(0.01)
    plt.cla()
    universe.step(0.05)
    xs = [balls[i].x[0] for i in range(len(balls))]
    ys = [balls[i].x[1] for i in range(len(balls))]
    plt.scatter(xs, ys)
    plt.scatter(balls[1].x[0], balls[1].x[1], color='r')
    plt.axis([0, 10, 0, 10])