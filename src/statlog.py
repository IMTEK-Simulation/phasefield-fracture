import os
import json
import numpy as np


def clear(fname):
    if os.path.exists(fname):
        os.remove(fname)

def dump(obj, fname):
    jsonfile = open(fname, mode='a+')
    json.dump(obj.__dict__,jsonfile,default=lambda o: "(array)")
    jsonfile.write('\n')
    jsonfile.close()

class timing():

    def __init__(self, fname):
        self.fname = fname
        # non-cumulative performance
        self.subiteration = 0
        self.phi_time = 0
        self.phi_subits = 0
        self.elastic_time = 0
        self.elastic_newton_its = 0
        self.elastic_CG_its = 0
        # cumulative performance
        self.subits_total = 0
        self.phi_time_total = 0
        self.phi_subits_total = 0
        self.elastic_time_total = 0
        self.elastic_newton_its_total = 0
        self.elastic_CG_its_total = 0
    
    def iteration_update(self):
        self.subiteration += 1
        self.subits_total += 1
        self.phi_time_total += self.phi_time
        self.phi_subits_total += self.phi_subits
        self.elastic_time_total += self.elastic_time
        self.elastic_newton_its_total += self.elastic_newton_its
        self.elastic_CG_its_total += self.elastic_CG_its


class stats():
    
    def __init__(self, fname, obj):
        self.fname = fname
        self.output_index = 0
        self.subit_output_index = 0
        self.output_flag = False
        self.subit_output_flag = False
        self.subiteration = 0
        # non-control-related data (2D specific)
        # after strain solve data
        self.max_overforce = 0
        self.coupling_at_ofmax = 0
        # delta phi data
        self.delta_phi = 0
        self.max_delta_phi = 0
        # stress data
        self.stress = 0
        self.avg_yy_stress = 0
        self.avg_xy_stress = 0
        self.avg_xx_stress = 0
        # obj, post-solve data
        self.strain = 0
        self.avg_coupling = 0
        self.max_coupling_en = 0
        self.max_coupling_jac = 0
        self.total_phi = obj.integrate(obj.phi.array())
        self.total_energy = obj.objective(obj.phi.array())
        self.total_elastic_energy = obj.integrate(obj.mechform.get_elastic_energy(obj))
        self.total_fracture_energy = obj.integrate(obj.fracture_energy(obj.phi.array()))
        self.delta_work_energy = 0
        self.total_work_energy = 0
        self.timestep = 0
        self.rescale_ratio = 0
        self.excess_energy = 0
        self.delta_energy = 0

    def get_stats_stress(self, stress, obj):
        stressyy = stress[3:stress.size:4]
        stressxy = stress[2:stress.size-1:4]
        stressxx = stress[0:stress.size-3:4]
        self.avg_yy_stress = obj.integrate(stressyy)/obj.domain_measure
        self.avg_xy_stress = obj.integrate(stressxy)/obj.domain_measure
        self.avg_xx_stress = obj.integrate(stressxx)/obj.domain_measure
        self.stress = max(self.avg_yy_stress, abs(self.avg_xy_stress), self.avg_xx_stress)
            
    def get_stats_elastic(self, obj):
        self.rescale_ratio = 0
        self.strain =  np.amax(obj.F_tot)
        coupling = obj.straineng.array()*obj.interp.energy(obj.phi.array())
        self.avg_coupling = obj.integrate(coupling)/obj.domain_measure
        self.max_coupling_en = obj.max(coupling)
        overforce = -obj.jacobian(obj.phi.array())
        couplingjac = -obj.straineng.array()*obj.interp.jac(obj.phi.array())
        self.max_coupling_jac = obj.max(couplingjac)
        self.max_overforce = obj.max(overforce)
        self.coupling_at_ofmax = obj.value_at_max(overforce, couplingjac)
        if (self.subiteration == 0):
            elastic_energy = obj.integrate(obj.mechform.get_elastic_energy(obj))
            self.delta_work_energy = elastic_energy - self.total_elastic_energy
            self.total_work_energy += self.delta_work_energy
        else:
            self.delta_work_energy = 0
            
    def get_stats_rescale(self, obj, ratio):
        self.strain =  np.amax(obj.F_tot)
        self.avg_yy_stress = self.avg_yy_stress*ratio
        self.avg_xy_stress = self.avg_xy_stress*ratio
        self.avg_xx_stress = self.avg_xx_stress*ratio
        self.stress = self.stress*ratio
        self.avg_coupling = self.avg_coupling*ratio**2
        self.max_coupling_en = self.max_coupling_en*ratio**2
        overforce = -obj.jacobian(obj.phi.array())
        couplingjac = -obj.straineng.array()*obj.interp.jac(obj.phi.array())
        self.max_coupling_jac = obj.max(couplingjac)
        self.max_overforce = obj.max(overforce)
        self.coupling_at_ofmax = obj.value_at_max(overforce, couplingjac)
        self.delta_work_energy = self.total_elastic_energy*ratio**2 - self.total_elastic_energy
        self.total_work_energy += self.delta_work_energy
        self.rescale_ratio = ratio

    def get_stats_phi(self, delta_phi, obj):
        self.delta_phi = obj.integrate(delta_phi)
        self.max_delta_phi = obj.max(delta_phi)
    
    def get_stats_full(self, obj):
        self.total_phi = obj.integrate(obj.phi.array())
        total_elastic_energy = obj.integrate(obj.mechform.get_elastic_energy(obj))
        delta_elastic = total_elastic_energy - self.total_elastic_energy
        self.total_elastic_energy = total_elastic_energy
        total_fracture_energy = obj.integrate(obj.fracture_energy(obj.phi.array()))
        delta_fracture = total_fracture_energy - self.total_fracture_energy
        self.total_fracture_energy = total_fracture_energy
        total_energy = obj.objective(obj.phi.array())
        self.delta_energy = total_energy - self.total_energy
        self.total_energy = total_energy
        self.excess_energy += delta_elastic + delta_fracture - self.delta_work_energy
        self.timestep = obj.dt
        self.subiteration += 1
        
    def subit_output_dump(self):
        self.subit_output_flag = True
        dump(self, self.fname)
        self.subit_output_flag = False
        self.subit_output_index += 1
        
    def output_dump(self):
        self.output_flag = True
        dump(self, self.fname)
        self.output_flag = False
        self.output_index += 1

