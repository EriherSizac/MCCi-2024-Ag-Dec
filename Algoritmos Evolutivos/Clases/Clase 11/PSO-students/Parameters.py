from numpy import number


class Parameters:
    def __init__(self, params_file):
        file = open(params_file, "rt")
        for line in file.readlines():
            fragments = line.split("=")
            if fragments[0] == "swarm_size":
                self.swarm_size = int(fragments[1])
            elif fragments[0] == "c1":
                self.c1 = float(fragments[1])
            elif fragments[0] == "c2":
                self.c2 = float(fragments[1])
            elif fragments[0] == "inertia_factor":
                self.w = float(fragments[1])
            elif fragments[0] == "nvar":
                self.nvar = int(fragments[1])
            elif fragments[0] == "Gmax":
                self.Gmax = int(fragments[1])    

    def get_swarm_size(self):
        return self.swarm_size

    def get_c1(self):
        return self.c1

    def get_c2(self):
        return self.c2
    
    def get_w(self):
        return self.w

    def get_nvar(self):
        return self.nvar 

    def get_Gmax(self):
        return self.Gmax
