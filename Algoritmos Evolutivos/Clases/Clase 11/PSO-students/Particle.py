import numpy as np

class Particle:
    def __init__(self, objective_function):
        self.__obj_func_singleton = objective_function
        self.x = np.empty(objective_function.get_nvar())       
        self.velocity = np.empty(objective_function.get_nvar())
        self.objective_value = None
    
    def get_x(self):
        return self.x
    
    def set_x(self, x):
        self.x = np.array(x, copy=True)
    
    def get_x_at(self, i):
        return self.x[i]

    def set_x_at(self, value, i):
        self.x[i] = value

    def get_velocity(self):
        return self.velocity
    
    def set_velocity(self, velocity):
        self.velocity = np.array(velocity, copy=True)
    
    def get_velocity_at(self, i):
        return self.velocity[i]
    
    def set_velocity_at(self, value, i):
        self.velocity[i] = value
    
    def get_objective_value(self):
        return self.objective_value
    
    def set_objective_value(self, objective_value):
        self.objective_value = objective_value
    
    def evaluate_objective_function(self):
        self.objective_value = self.__obj_func_singleton.evaluate(self.x)
    
    def initialize_location(self, value=None):
        if value is None:
            self.x = np.empty(self.__obj_func_singleton.get_nvar())
            xmin = self.__obj_func_singleton.get_xmin()
            xmax = self.__obj_func_singleton.get_xmax()
            for i in range(self.__obj_func_singleton.get_nvar()):
                self.x[i] = xmin[i] + np.random.rand() * (xmax[i] - xmin[i])
            self.objective_value = self.__obj_func_singleton.evaluate(self.x)
        else:
            self.x = np.full(self.__obj_func_singleton.get_nvar(), np.infty)
            self.objective_value = value
    
class ParticleFactory:
    def __init__(self, obj_func_singleton):
        self.__obj_func_singleton = obj_func_singleton

    def create_particle(self):
        return Particle(self.__obj_func_singleton)