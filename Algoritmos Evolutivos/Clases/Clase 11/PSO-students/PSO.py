import numpy as np
from Swarm import Swarm
from Particle import ParticleFactory

class PSO:
    def __init__(self, params, obj_func):
        self.params = params
        self.obj_func = obj_func
        self.particle_factory = ParticleFactory(obj_func)       
        self.swarm = Swarm(params.get_swarm_size(), self.particle_factory)
        self.lbest = Swarm(params.get_swarm_size(), self.particle_factory)
        self.c1 = self.params.get_c1()
        self.c2 = self.params.get_c2()
        self.w = self.params.get_w()
    
    def run(self, execution, report):
        self.swarm.initialize_swarm()
        self.lbest.initialize_lbest_swarm()
        gbest = self.particle_factory.create_particle()
        gbest.initialize_location(np.infty)
        t = 0
        delta = np.random.random()
        while t < self.params.get_Gmax():
            for i in range(self.swarm.get_swarm_size()):
                particle = self.swarm.get_particle_at(i)
                particle_lbest = self.lbest.get_particle_at(i)
                # Set the personal best position
                if particle.get_objective_value() < particle_lbest.get_objective_value():
                    particle_lbest.set_x(particle.get_x())
                    particle_lbest.set_objective_value(particle.get_objective_value())
                    
                # Update the gBest position
                if particle_lbest.get_objective_value() < gbest.get_objective_value():
                    gbest.set_x(particle_lbest.get_x())
                    gbest.set_objective_value(particle_lbest.get_objective_value())
                    
            # Random numbers for the calculation of the velocity
            r1 = np.random.rand(self.obj_func.get_nvar(), 1)
            r2 = np.random.rand(self.obj_func.get_nvar(), 1)

            # For each particle, update its velocity and position
            for i in range(self.swarm.get_swarm_size()):
                particle = self.swarm.get_particle_at(i)
                lbest = self.lbest.get_particle_at(i)
                for j in range(self.obj_func.get_nvar()):
                    # Update velocity and position of each particle
                    max_velocity = delta*(self.swarm.get_particle_at(-1).get_x_at(j) - self.swarm.get_particle_at(0).get_x_at(j))
                    velocity = particle.get_velocity_at(j) + self.c1*r1[j][0]*(lbest.get_x_at(j) - particle.get_x_at(j)) + self.c2*r2[j][0]*(gbest.get_x_at(j) - particle.get_x_at(j))
                    if velocity >= max_velocity:
                        velocity = max_velocity
                    particle.set_velocity_at(velocity, j)

                    value = particle.get_x_at(j) + velocity
                    particle.set_x_at(value, j) # value is the new position for the j-th component of the particle
                particle.evaluate_objective_function() # Calculate the objective value based on the new position of the particle
            report.add_best_individual_at_generation(t, execution, gbest) # Do not modify!
            t = t + 1
        return gbest


