import numpy as np
import random

class particle:
    def __init__(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.pBest_position = np.array(position)
        self.pBest_value = float('inf')
    
    def update_velocity(self, w, c1, c2, gBest_position) :
        r1 = np.random.random(self.position.shape)
        r2 = np.random.random(self.position.shape)
        cognitive_velocity = c1*r1 * (self.pBest_position - self.position)
        social_velocity = c2 * r2 * (gBest_position - self.position)

        self.velocity = w * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds ) :
        self.position = self.position + self.velocity
        self.position = np.maximum(self.position, bounds[0])
        self.position = np.minimum(self.position, bounds[1])

    class Swarm:
        def __init__(self, num_particle, bounds, function, w = 0.5, c1 = 1, c2 = 2):
            self.num_particle = num_particle
            self.bounds = bounds
            self.w = w
            self.c1 = c1
            self.c2 = c2
            self.gBest_position = None
            self.gBest_value = float('inf')

        def optimize (self, max_iterations) :
            fx = 
            if (fx < )      