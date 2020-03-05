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
    
    def is_moving_closer(self, ball):
        n = (ball.x - self.x)
        v = ball.v - self.v
        return np.dot(v, n) < 0
    
    def move(self, dt, F=np.zeros(3), g=np.zeros(3)):
        F_net = F + self.m*g
        self.act_force(F_net, dt)
        
    def act_force(self, F, dt):
        a = F/self.m
        x_new = self.x + 0.5*a*dt**2 + self.v*dt
        v_new = self.v + a*dt
        
        self.x = x_new
        self.v = v_new
    
    def has_collided_ball(self, ball):
        return np.sum((self.x - ball.x)**2) <= (self.r + ball.r)**2
    
    def has_collided_box(self, box):
        hit_outer = np.any(self.x + self.r >= box.dims)
        hit_inner = np.any(self.x - self.r <= np.zeros(3))
        return hit_outer or hit_inner


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
    def has_pair_moving_closer(balls_pairs):
        for ball1, ball2 in balls_pairs:
            if ball1.is_moving_closer(ball2):
                return True
        return False
    
    @staticmethod
    def who_collided_balls(balls):
        n = len(balls)
        collided = []
        for i in range(n-1):
            for j in range(i+1, n):
                if Universe.has_collided_ball_ball(balls[i], balls[j]):
                    collided.append((balls[i], balls[j]))
        return collided
    
    @staticmethod
    def who_collided_box(balls, box):
        collided = []
        for ball in balls:
            if Universe.has_collided_ball_box(ball, box):
                collided.append(ball)
        return collided
    
    @staticmethod
    def collide_ball_ball(ball1, ball2, e=1):
        m1, m2 = ball1.m, ball2.m
        x1, x2 = ball1.x, ball2.x
        v1, v2 = ball1.v, ball2.v
        
        M = m1 + m2
        x_rel = x1 - x2
        n = x_rel/np.sqrt(np.sum(x_rel**2))
        
        vM = 1/M * (m1*v1 + m2*v2)
        
        v1M = v1 - vM
        v2M = v2 - vM
        v1M_new = v1M - (1 + e)*v1M.dot(n)*n
        v2M_new = v2M - (1 + e)*v2M.dot(n)*n
        
        ball1.v = v1M_new + vM
        ball2.v = v2M_new + vM
        
    @staticmethod
    def collide_ball_box(ball, box, e=1):
        hit_outer = (ball.x + ball.r >= box.dims)
        hit_inner = (ball.x - ball.r <= np.zeros(3))
        
        ball.v[hit_outer] *= -e
        ball.v[hit_inner] *= -e
        
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
    
    @staticmethod
    def collide_resolve(balls, box, dt, Fs, g):
        balls_virtual = Universe.virtual_step(balls, box, dt, Fs, g)
        collided_pairs = Universe.who_collided_balls(balls_virtual)
        collided_box = Universe.who_collided_box(balls_virtual, box)
        
        for ball1, ball2 in collided_pairs:
            Universe.collide_ball_ball(ball1, ball2)
        
        for ball in collided_box:
            Universe.collide_ball_box(balls, box)
    
    @staticmethod
    def move(balls, dt, Fs, g):
        for ball, F in zip(balls, Fs):
            ball.move(dt, F, g)
    
    @staticmethod
    def virtual_step(balls, box, dt, Fs, g):
        balls_virtual = Universe.copy_balls(balls)
        Universe.move(balls_virtual, dt, Fs, g)
        return balls_virtual
    
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
        
        if Universe.has_collision(balls, box):
            Universe.collide(balls, box)
        
        if Universe.has_collision_balls_balls(balls):
            collided_pairs = Universe.who_collided_balls(balls)
            while Universe.has_pair_moving_closer(collided_pairs):
                print('resolve')
                for ball1, ball2 in collided_pairs:
                    if ball1.is_moving_closer(ball2):
                        Universe.collide_ball_ball(ball1, ball2)
        
        if Universe.has_collision_balls_box(balls, box):
            pass
            
        #collided_pairs = Universe.who_collided_balls(balls)
        #collided_box = Universe.who_collided_box(balls, box)
        
        #while Universe.will_collide(balls, box, dt, Fs_next, g):
            #balls_prev = Universe.copy_balls(balls)
            #Universe.collide_resolve(balls, box, dt, Fs_next, g)
            #if Universe.same_balls_states(balls, balls_prev):
            #    break