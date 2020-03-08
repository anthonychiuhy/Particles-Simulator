# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:40:01 2020

@author: Anthony
"""
import numpy as np

class Ball:
    def __init__(self, x=None, v=None, m=1, r=1):
        if x is None:
            x = np.zeros(3)
        if v is None:
            v = np.zeros(3)
        
        self.x = x
        self.v = v
        self.m = m
        self.r = r
    
    def copy(self):
        x = self.x.copy()
        v = self.v.copy()
        m = self.m
        r = self.r
        return Ball(x, v, m, r)
    
    def is_colliding_ball(self, ball):
        return np.sum((self.x - ball.x)**2) <= (self.r + ball.r)**2
    
    def is_colliding_box(self, box):
        colliding_outer = np.any(self.x + self.r >= box.dims)
        colliding_inner = np.any(self.x - self.r <= np.zeros(3))
        return colliding_outer or colliding_inner
    
    def is_moving_closer_ball(self, ball):
        n = (ball.x - self.x)
        v = ball.v - self.v
        return np.dot(v, n) < 0
    
    def is_collapsing_into_ball(self, ball):
        return self.is_colliding_ball(ball) and self.is_moving_closer_ball(ball)
    
    def is_collapsing_into_box(self, box):
        colliding_outer = (self.x + self.r >= box.dims)
        colliding_inner = (self.x - self.r <= np.zeros(3))
        
        moving_closer_outer = self.v > 0
        moving_closer_inner = self.v < 0
        
        collapsing_outer = np.any(colliding_outer & moving_closer_outer)
        collapsing_inner = np.any(colliding_inner & moving_closer_inner)
        
        return collapsing_outer or collapsing_inner
    
    def collide_ball(self, ball, e=1):
        m1, m2 = self.m, ball.m
        x1, x2 = self.x, ball.x
        v1, v2 = self.v, ball.v
        
        M = m1 + m2
        x_rel = x1 - x2
        n = x_rel/np.sqrt(np.sum(x_rel**2))
        
        vM = 1/M * (m1*v1 + m2*v2)
        
        v1M = v1 - vM
        v2M = v2 - vM
        v1M_new = v1M - (1 + e)*v1M.dot(n)*n
        v2M_new = v2M - (1 + e)*v2M.dot(n)*n
        
        self.v = v1M_new + vM
        ball.v = v2M_new + vM
    
    def collide_box(self, box, e=1):
        colliding_outer = (self.x + self.r >= box.dims)
        colliding_inner = (self.x - self.r <= np.zeros(3))
        
        moving_closer_outer = self.v > 0
        moving_closer_inner = self.v < 0
        
        collapsing_outer = colliding_outer & moving_closer_outer
        collapsing_inner = colliding_inner & moving_closer_inner
        
        self.v[collapsing_outer] *= -e
        self.v[collapsing_inner] *= -e
        
    def act_force(self, F, dt):
        a = F/self.m
        x_new = self.x + 0.5*a*dt**2 + self.v*dt
        v_new = self.v + a*dt
        
        self.x = x_new
        self.v = v_new
        
    def move(self, dt, F=np.zeros(3), g=np.zeros(3)):
        F_net = F + self.m*g
        self.act_force(F_net, dt)


class Box:
    def __init__(self, width, height, depth):
        dims = np.array([width, height, depth])
        self.dims = dims


class Universe:
    def __init__(self, box, balls, g=np.zeros(3)):
        self.balls = balls
        self.box = box
        self.g = g
    
    @staticmethod    
    def has_colliding_balls_balls(balls):
        n = len(balls)
        for i in range(n-1):
            for j in range(i+1, n):
                if balls[i].is_colliding_ball(balls[j]):
                    return True
        return False
        
    @staticmethod
    def has_colliding_balls_box(balls, box):
        for ball in balls:
            if ball.is_colliding_box(box):
                return True
        return False
    
    @staticmethod
    def has_colliding(balls, box):
        return Universe.has_colliding_balls_balls(balls) or Universe.has_colliding_balls_box(balls, box)
    
    @staticmethod
    def has_collapsing_balls_balls(balls):
        n = len(balls)
        for i in range(n-1):
            for j in range(i+1, n):
                if balls[i].is_collapsing_into_ball(balls[j]):
                    return True
        return False
                
    @staticmethod
    def has_collapsing_balls_box(balls, box):
        for ball in balls:
            if ball.is_collapsing_into_box(box):
                return True
        return False
    
    @staticmethod
    def has_collapsing(balls, box):
        return Universe.has_collapsing_balls_balls(balls) or Universe.has_collapsing_balls_box(balls, box)
    
    @staticmethod
    def find_colliding_balls_balls(balls):
        n = len(balls)
        colliding = []
        for i in range(n-1):
            for j in range(i+1, n):
                if balls[i].is_colliding_ball(balls[j]):
                    colliding.append((balls[i], balls[j]))
        return colliding
    
    @staticmethod
    def find_colliding_balls_box(balls, box):
        colliding = []
        for ball in balls:
            if ball.is_colliding_box(box):
                colliding.append(ball)
        return colliding
    
    @staticmethod
    def find_collapsing_balls_balls(balls):
        n = len(balls)
        collapsing = []
        for i in range(n-1):
            for j in range(i+1, n):
                if balls[i].is_collapsing_into_ball(balls[j]):
                    collapsing.append((balls[i], balls[j]))
        return collapsing
    
    @staticmethod
    def find_collapsing_balls_box(balls, box):
        collapsing = []
        for ball in balls:
            if ball.is_collapsing_into_box(box):
                collapsing.append(ball)
        return collapsing
    
    @staticmethod
    def collide_resolve(balls, box):
        n = len(balls)
        
        for i in range(n-1):
            for j in range(i+1, n):
                if balls[i].is_collapsing_into_ball(balls[j]):
                    balls[i].collide_ball(balls[j])
        
        for i in range(n):
            balls[i].collide_box(box)
    
    @staticmethod
    def move(balls, dt, Fs, g):
        for ball, F in zip(balls, Fs):
            ball.move(dt, F, g)
    
    def step(self, dt, Fs=None):
        balls = self.balls
        box = self.box
        g = self.g
        if Fs is None:
            Fs = np.zeros((len(balls), 3))
        
        # Advance time step
        Universe.move(balls, dt, Fs, g)
        
        # Resolve collisions
        while Universe.has_collapsing(balls, box):
            Universe.collide_resolve(balls, box)