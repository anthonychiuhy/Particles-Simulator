# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:38:00 2020

@author: Anthony
"""

from objects import Universe, Box, Ball

import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

box = Box(10, 10, 10)

x_1, v_1 = np.array([2,5,5]), np.array([2,0,0])
x_2, v_2 = np.array([5,5,5]), np.array([0,0,0])
x_3, v_3 = np.array([7,5,5]), np.array([0,0,0])
x_4, v_4 = np.array([9,5,5]), np.array([0,0,0])

b1 = Ball(x_1, v_1, r=1, m=1)
b2 = Ball(x_2, v_2, r=1, m=3)
b3 = Ball(x_3, v_3, r=1, m=1)
b4 = Ball(x_4, v_4, r=1, m=1)

balls = [b1, b2, b3, b4]

g = np.array([0,0,0])
universe = Universe(box, balls, g)

plt.subplots(1,1)


for i in range(200):
    plt.pause(0.01)
    plt.cla()
    universe.step(0.2)
    xs = [balls[i].x[0] for i in range(len(balls))]
    ys = [balls[i].x[1] for i in range(len(balls))]
    plt.scatter(xs, ys)
    plt.axis([0, 10, 0, 10])
    
    print(b1.v[0], b2.v[0], b3.v[0], b1.has_collided_ball(b2), b3.has_collided_ball(b2))