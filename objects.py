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
    
    def same_state_as(self, ball):
        same_x = np.all(self.x == ball.x)
        same_v = np.all(self.v == ball.v)
        same_m = self.m == ball.m
        same_r = self.r == ball.r
        return same_x and same_v and same_m and same_r
    
    def has_collided_ball(self, ball):
        return np.sum((self.x - ball.x)**2) <= (self.r + ball.r)**2
    
    def has_collided_box(self, box):
        hit_outer = np.any(self.x + self.r >= box.dims)
        hit_inner = np.any(self.x - self.r <= np.zeros(3))
        return hit_outer or hit_inner
    
    def is_moving_closer_ball(self, ball):
        n = (ball.x - self.x)
        v = ball.v - self.v
        return np.dot(v, n) < 0
     
    def is_collapsing_into_ball(self, ball):
        return self.has_collided_ball(ball) and self.is_moving_closer_ball(ball)
    
    def is_collapsing_into_box(self, box):
        hit_outer = (self.x + self.r >= box.dims)
        hit_inner = (self.x - self.r <= np.zeros(3))
        
        moving_into_outer = np.any(hit_outer * self.v > 0)
        moving_into_inner = np.any(hit_inner * self.v < 0)
        
        return moving_into_inner or moving_into_outer
    
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
        hit_outer = (self.x + self.r >= box.dims)
        hit_inner = (self.x - self.r <= np.zeros(3))
        
        self.v[hit_outer] *= -e
        self.v[hit_inner] *= -e
    
    def move(self, dt, F=np.zeros(3), g=np.zeros(3)):
        F_net = F + self.m*g
        self.act_force(F_net, dt)
        
    def act_force(self, F, dt):
        a = F/self.m
        x_new = self.x + 0.5*a*dt**2 + self.v*dt
        v_new = self.v + a*dt
        
        self.x = x_new
        self.v = v_new
    

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
    def copy_balls(balls):
        balls_copy = []
        for ball in balls:
            balls_copy.append(ball.copy())
        return balls_copy
    
    @staticmethod
    def same_balls_states(balls1, balls2):
        same = []
        for ball1, ball2 in zip(balls1, balls2):
            same.append(ball1.same_state_as(ball2))
        return all(same)
    
    @staticmethod
    def has_collided_ball_ball(ball1, ball2):
        return ball1.has_collided_ball(ball2)
    
    @staticmethod
    def has_collided_ball_box(ball, box):
        return ball.has_collided_box(box)
    
    @staticmethod    
    def has_collision_balls_balls(balls):
        n = len(balls)
        for i in range(n-1):
            for j in range(i+1, n):
                if Universe.has_collided_ball_ball(balls[i], balls[j]):
                    return True
        return False
        
    @staticmethod
    def has_collision_balls_box(balls, box):
        for ball in balls:
            if Universe.has_collided_ball_box(ball, box):
                return True
        return False
    
    @staticmethod
    def has_collision(balls, box):
        return Universe.has_collision_balls_balls(balls) or Universe.has_collision_balls_box(balls, box)
    
    @staticmethod
    def has_balls_collapsing(balls):
        n = len(balls)
        for i in range(n-1):
            for j in range(i+1, n):
                if balls[i].is_collapsing_into_ball(balls[j]):
                    return True
        return False
                
    @staticmethod
    def has_ball_box_collapsing(balls, box):
        for ball in balls:
            if ball.is_collapsing_into_box(box):
                return True
        return False
    
    @staticmethod
    def has_collapse(balls, box):
        return Universe.has_balls_collapsing(balls) or Universe.has_ball_box_collapsing(balls, box)
    
    @staticmethod
    def find_collided_pairs(balls):
        n = len(balls)
        collided = []
        for i in range(n-1):
            for j in range(i+1, n):
                if Universe.has_collided_ball_ball(balls[i], balls[j]):
                    collided.append((balls[i], balls[j]))
        return collided
    
    @staticmethod
    def find_collided_balls_box(balls, box):
        collided = []
        for ball in balls:
            if Universe.has_collided_ball_box(ball, box):
                collided.append(ball)
        return collided
    
    @staticmethod
    def find_collapsing_pairs(balls):
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
    def collide_ball_ball(ball1, ball2, e=1):
        ball1.collide_ball(ball2)
        
    @staticmethod
    def collide_ball_box(ball, box, e=1):
        ball.collide_box(box)
    
    """
    @staticmethod
    def collide_balls_balls(balls):
        n = len(balls)
        for i in range(n-1):
            for j in range(i+1, n):
                if Universe.has_collided_ball_ball(balls[i], balls[j]):
                    Universe.collide_ball_ball(balls[i], balls[j])
    
    @staticmethod
    def collide_balls_box(balls, box):
        for ball in balls:
            if Universe.has_collided_ball_box(ball, box):
                Universe.collide_ball_box(ball, box)
    
    
    @staticmethod
    def will_collide(balls, box, dt, Fs, g):
        balls_virtual = Universe.virtual_step(balls, box, dt, Fs, g)
        return Universe.has_collision_balls_balls(balls_virtual) or Universe.has_collision_balls_box(balls_virtual, box)
    
    @staticmethod
    def collide(balls, box):
        Universe.collide_balls_balls(balls)
        Universe.collide_balls_box(balls, box)
    """
    
    @staticmethod
    def collide_resolve(balls, box):
        print('resolve')
        collided_pairs = Universe.find_collapsing_pairs(balls)
        collided_box = Universe.find_collapsing_balls_box(balls, box)
        for ball1, ball2 in collided_pairs:
            ball1.collide_ball(ball2)
        
        for ball in collided_box:
            ball.collide_box(box)
    
    @staticmethod
    def move(balls, dt, Fs, g):
        for ball, F in zip(balls, Fs):
            ball.move(dt, F, g)
    
    """
    @staticmethod
    def virtual_step(balls, box, dt, Fs, g):
        balls_virtual = Universe.copy_balls(balls)
        Universe.move(balls_virtual, dt, Fs, g)
        return balls_virtual
    """
    
    def step(self, dt, Fs=None, Fs_next=None):
        balls = self.balls
        box = self.box
        g = self.g
        if Fs is None:
            Fs = np.zeros((len(balls), 3))
        if Fs_next is None:
            Fs_next = np.zeros((len(balls), 3))
            
        # Advance time step
        Universe.move(balls, dt, Fs, g)
        
        
        #collided_pairs = Universe.find_collided_pairs(balls)
        #collided_box_balls = Universe.find_collided_box_balls(balls, box)
        
        while Universe.has_collapse(balls, box):
            Universe.collide_resolve(balls, box)
            
        
        #if Universe.has_collision(balls, box):
        #    Universe.collide(balls, box)
        
        #if Universe.has_collision_balls_balls(balls):
        #    collided_pairs = Universe.who_collided_balls(balls)
        #    while Universe.has_pair_moving_closer(collided_pairs):
        #        print('resolve')
        #       for ball1, ball2 in collided_pairs:
        #           if ball1.is_moving_closer(ball2):
        #                Universe.collide_ball_ball(ball1, ball2)
        
        #if Universe.has_collision_balls_box(balls, box):
        #    pass
            
        #collided_pairs = Universe.who_collided_balls(balls)
        #collided_box = Universe.who_collided_box(balls, box)
        
        #while Universe.will_collide(balls, box, dt, Fs_next, g):
            #balls_prev = Universe.copy_balls(balls)
            #Universe.collide_resolve(balls, box, dt, Fs_next, g)
            #if Universe.same_balls_states(balls, balls_prev):
            #    break