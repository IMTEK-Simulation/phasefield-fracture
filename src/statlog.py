import os
import json

class statdump():

    def __init__(self, fname):
        self.fname = fname
    
    def clear(self):
        if os.path.exists(self.fname):
            os.remove(self.fname)

    def dump(self):
        jsonfile = open(self.fname, mode='a+')
        json.dump(self.__dict__,jsonfile,default=lambda o: "(array)")
        jsonfile.write('\n')
        jsonfile.close()


class full_iteration_stats(statdump):
    
    def __init__(self, fname):
        statdump.__init__(self, fname)
        self.avg_strain = 0
        self.total_energy = 0
        self.coupling_energy = 0
        self.delta_phi = 0
        self.subiterations = 0
        self.elastic_newton_its = 0
        self.elastic_CG_its = 0
        self.elastic_time = 0
        self.phi_subits = 0
        self.phi_time = 0

    def subiteration_update(self, altmin_obj):
        self.subiterations += 1
        self.elastic_time += altmin_obj.elastic_time
        self.phi_time += altmin_obj.phi_time
        self.elastic_newton_its += altmin_obj.elastic_newton_its
        self.elastic_CG_its += altmin_obj.elastic_CG_its
        self.phi_subits += altmin_obj.phi_subits
        
    def iteration_reset(self):
        self.subiterations = 0
        self.elastic_subits = 0
        self.elastic_time = 0
        self.phi_subits = 0
        self.phi_time = 0
        
class altmin_iteration_stats(statdump):

    def __init__(self, fname, avg_strain):
        statdump.__init__(self,fname)
        self.avg_strain = avg_strain
        self.subiteration = 0
        self.elastic_time = 0
        self.elastic_newton_its = 0
        self.elastic_CG_its = 0
        self.phi_time = 0
        self.phi_subits = 0
        self.delta_energy = 0
        self.delta_phi = 0
        
        
    
