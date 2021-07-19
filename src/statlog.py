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


class stats():
    
    def __init__(self, fname):
        self.fname = fname
        self.output_index = 0
        self.subit_output_index = 0
        self.output_flag = False
        self.subit_output_flag = False
        
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
        # non-control-related data (2D specific)
        # after strain solve data
        self.delta_elastic_elastic = 0
        self.avg_overforce = 0
        self.max_overforce = 0
        self.coupling_at_ofmax = 0
        # delta phi data
        self.delta_phi = 0
        self.max_delta_phi = 0
        self.max_delta_phix = 0
        self.max_delta_phiy = 0
        self.delta_elastic_phi = 0
        # stress data
        self.stress = 0
        self.avg_yy_stress = 0
        self.avg_xy_stress = 0
        self.avg_xx_stress = 0
        self.max_yy_stress = 0
        self.max_xy_stress = 0
        self.max_xx_stress = 0
        # obj, post-solve data
        self.strain = 0
        self.avg_yy_strain = 0
        self.avg_xy_strain = 0
        self.avg_xx_strain = 0
        self.max_yy_strain = 0
        self.max_xy_strain = 0
        self.max_xx_strain = 0
        self.avg_coupling = 0
        self.max_coupling_en = 0
        self.max_coupling_jac = 0
        self.max_couplingx = 0
        self.max_couplingy = 0
        self.total_phi = 0
        self.total_energy = 0
        self.total_elastic_energy = 0
        self.total_fracture_energy = 0
        self.total_energy_lastiteration = 0
        self.timestep = 0
        self.elastic_scale_factor = 1.0
        
    def get_stats_stress(self, stress, obj):
        stressyy = stress[3:stress.size:4]
        stressxy = stress[2:stress.size-1:4]
        stressxx = stress[0:stress.size-3:4]
        self.avg_yy_stress = obj.integrate(stressyy)/obj.domain_measure
        self.avg_xy_stress = obj.integrate(stressxy)/obj.domain_measure
        self.avg_xx_stress = obj.integrate(stressxx)/obj.domain_measure
        self.stress = max(self.avg_yy_stress, self.avg_xy_stress, self.avg_xx_stress)
        self.max_yy_stress = obj.max(stressyy)
        self.max_xy_stress = obj.max(stressxy)
        self.max_xx_stress = obj.max(stressxx)
            
    def get_stats_elastic(self, obj):
        self.avg_yy_strain = obj.F_tot[1,1]
        self.avg_xy_strain = (obj.F_tot[0,1] + obj.F_tot[1,0])/2
        self.avg_xx_strain = obj.F_tot[0,0]
        self.strain =  max(self.avg_yy_strain, self.avg_xy_strain,self.avg_xx_strain)
        self.max_yy_strain = obj.max(obj.strain.array()[3,0,:,:])
        self.max_xy_strain = obj.max(obj.strain.array()[2,0,:,:])
        self.max_xx_strain = obj.max(obj.strain.array()[0,0,:,:])
        coupling = obj.straineng.array()*obj.interp.energy(obj.phi.array())
        self.avg_coupling = obj.integrate(coupling)/obj.domain_measure
        self.max_coupling_en = obj.max(coupling)
        self.max_couplingx, self.max_couplingy = obj.argmax(coupling)
        overforce = -obj.jacobian(obj.phi.array())
        couplingjac = -obj.straineng.array()*obj.interp.jac(obj.phi.array())
        self.max_coupling_jac = obj.max(couplingjac)
        self.avg_overforce = obj.integrate(overforce)/obj.domain_measure
        self.max_overforce = obj.max(overforce)
        self.coupling_at_ofmax = obj.value_at_max(overforce, couplingjac)
        self.delta_elastic_elastic =  obj.objective(obj.phi.array()) - self.total_energy
        
    def get_stats_phi(self, delta_phi, obj):
        self.delta_phi = obj.integrate(delta_phi)
        self.max_delta_phi = obj.max(delta_phi)
        self.max_delta_phix, self.max_delta_phiy = obj.argmax(delta_phi)
    
    def get_stats_full(self, obj):
        self.total_phi = obj.integrate(obj.phi.array())
        elastic = obj.mechform.get_elastic_energy(obj)
        self.delta_elastic_phi = (self.total_elastic_energy - obj.integrate(elastic)
                - self.delta_elastic_elastic)
        self.total_elastic_energy = obj.integrate(elastic)
        energy = obj.energy_density(obj.phi.array())
        self.total_fracture_energy = obj.integrate(energy - elastic)
        total_energy = obj.integrate(energy)
        self.delta_energy = total_energy - self.total_energy
        self.total_energy = total_energy
        self.timestep = obj.dt
        
    def iteration_update(self):
        self.subiteration += 1
        self.subits_total += 1
        self.phi_time_total += self.phi_time
        self.phi_subits_total += self.phi_subits
        self.elastic_time_total += self.elastic_time
        self.elastic_newton_its_total += self.elastic_newton_its
        self.elastic_CG_its_total += self.elastic_CG_its
        
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
        
        
    
