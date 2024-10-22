import numpy as np

from abc import ABCMeta, abstractmethod
class ObjectiveFunction(metaclass=ABCMeta):
    def __init__(self, nvar):
        self.nvar = nvar
        self.xmin = np.empty(nvar)
        self.xmax = np.empty(nvar)
        self.set_xmin()
        self.set_xmax()
        
    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def set_xmin(self):
        pass
    
    @abstractmethod
    def set_xmax(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_nvar(self):
        return self.nvar

    def get_xmin(self):
        return self.xmin

    def get_xmin_at(self, index):
        return self.xmin[index]
    
    def get_xmax(self):
        return self.xmax
    
    def get_xmax_at(self, index):
        return self.xmax[index]
   

class sphere(ObjectiveFunction):        
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar):
            result = result + x[i] ** 2
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -5.0

    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 5.0
    
    def get_name(self):
        return sphere.__name__
   

class rastringin(ObjectiveFunction):     
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar):
            result = result + x[i]*x[i] - 10*np.cos(2*np.pi*x[i])
        result = result + 10*self.nvar
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -5.12
    
    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 5.12
    
    def get_name(self):
        return rastringin.__name__

class rosenbrock(ObjectiveFunction):       
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar - 1):
            result = result + 100*np.power(x[i + 1] - x[i]*x[i], 2) + np.power(1 - x[i], 2)
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -10.0
    
    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 10.0
    
    def get_name(self):
        return rosenbrock.__name__

class FunctionFactory:
    function_dictionary = {sphere.__name__: sphere,
                            rastringin.__name__: rastringin,
                            rosenbrock.__name__: rosenbrock}
    @classmethod
    def select_function(cls, function_name, nvar):
        return cls.function_dictionary[function_name](nvar)