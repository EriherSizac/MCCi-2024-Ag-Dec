import numpy as np
from Particle import Particle

class Swarm:
    def __init__(self, swarm_size, particle_factory):
        self.swarm_size = swarm_size
        self.particle_factory = particle_factory
        self.swarm = np.empty(swarm_size, dtype=Particle)

    def get_swarm_size(self):
        return self.swarm_size  
    
    def add_particle_at(self, index, particle):
        self.swarm[index] = particle
    
    def get_particle_at(self, index):
        return self.swarm[index]
    
    def initialize_lbest_swarm(self):
        for i in range(self.swarm_size):
            particle = self.particle_factory.create_particle()
            particle.initialize_location(np.infty)
            self.add_particle_at(i, particle)
    
    def initialize_swarm(self):
        for i in range(self.swarm_size):
            particle = self.particle_factory.create_particle()
            particle.initialize_location()
            self.add_particle_at(i, particle)