import mechanics
import model_components
import parallel_fracture
import numpy as np
import json
import time
import statlog


class simulation():
    
    def __init__(self, obj):
        self.obj = obj
        self.subit_outputname = 'altmin_subit.nc'
        self.fullit_outputname = 'test.nc'
        self.subit_statsname = 'stats_subit.json'
        self.fullit_statsname = 'stats_fullit.json'
        self.stats = statlog.full_iteration_stats(self.fullit_statsname)
        dummy_subit_stats = statlog.altmin_iteration_stats(self.subit_statsname, 0.00)
        if(obj.comm.rank == 0): 
            self.stats.clear()
            dummy_subit_stats.clear()
        self.domain_measure = np.array(obj.lens).prod()
        self.delta_energy_tol = 1e-6*self.domain_measure
        self.total_energy = self.obj.objective(self.obj.phi.array())
        self.delta_energy = 0.0
        self.energy_old = 0.0
        self.strain_step_tensor = np.array([[0,0],[0,1.0]])
        self.strain_step_scalar = 0.008
        
    def avg_strain(self):
        return np.max(np.linalg.eigvals(self.obj.F_tot))
        
    def subiteration(self):
        subit_stats = statlog.altmin_iteration_stats(self.subit_statsname, self.avg_strain())
        self.energy_old = self.total_energy
        start = time.time() 
        strain_solve = self.obj.strain_solver()
        straint = time.time()
        self.obj.mechform.get_elastic_coupling(self.obj, strain_solve)
        phi_solve = self.obj.phi_solver() 
        self.obj.phi.array()[...] += phi_solve.result
        self.total_energy = self.obj.objective(self.obj.phi.array())
        phit = time.time()
        subit_stats.delta_energy = self.total_energy - self.energy_old
        self.delta_energy = abs(subit_stats.delta_energy)
        subit_stats.elastic_subits = strain_solve.nb_it
        subit_stats.elastic_time = straint-start
        subit_stats.phi_subits = phi_solve.n_iterations
        subit_stats.phi_time = phit-straint
        self.stats.subiteration_update(subit_stats)
        if(self.obj.comm.rank == 0):
            subit_stats.dump()
        
    def iteration(self):
        self.delta_energy = 1.0
        delta_energy_old = 1e8
        self.obj.muOutput(self.subit_outputname,new=True)   
        while(self.delta_energy > self.delta_energy_tol):
            self.subiteration()
            if(self.obj.comm.rank == 0):
                print('delta energy = ', self.delta_energy)
            if((self.delta_energy > delta_energy_old + self.delta_energy_tol)
                   and (self.strain_step_scalar > 0.0005) and (self.stats.subiterations > 1)):
                self.obj.F_tot -= self.strain_step_tensor*self.strain_step_scalar
                self.strain_step_scalar /= 2
                self.obj.F_tot += self.strain_step_tensor*self.strain_step_scalar
                self.stats.subiterations = 1
                delta_energy_old = 1e8
                self.obj.phi.array()[...] = self.obj.phi_old
                if (self.obj.comm.rank == 0):
                    print('non-monotonicity of energy convergence detected, strain step reduced to ', 
                      self.strain_step_scalar)
            else:
                delta_energy_old = self.delta_energy
            if((self.stats.subiterations % 20 == 0) 
              or ((self.delta_energy > 0.000025*self.domain_measure*self.obj.Young) 
              and (self.stats.subiterations > 1))):
                if(self.obj.comm.rank == 0):
                    print('saving subiteration # ', self.stats.subiterations)
                self.obj.muOutput(self.subit_outputname)

    def run_simulation(self):
        nmax = 50
        self.obj.F_tot = self.strain_step_scalar*self.strain_step_tensor
        self.obj.phi_old = self.obj.phi.array() + 0.0
        self.obj.muOutput(self.fullit_outputname,new=True)
        self.stats = statlog.full_iteration_stats(self.fullit_statsname)
        n = 0
        while (n < nmax):
            self.iteration()
            self.stats.avg_strain = self.avg_strain()
            self.stats.total_energy = self.total_energy
            self.stats.delta_phi = self.obj.integrate(self.obj.phi.array()-self.obj.phi_old)
            self.stats.coupling_energy = self.obj.integrate(self.obj.straineng*
                self.obj.interp.energy(self.obj.phi.array()))

            self.obj.phi_old = np.maximum(self.obj.phi.array(),self.obj.phi_old)
            if(self.obj.comm.rank == 0):
                print('strain: ', self.stats.avg_strain, 'energy: ',
                    self.stats.total_energy,'delta phi',self.stats.delta_phi)
                self.stats.dump()
            self.obj.muOutput(self.fullit_outputname)
            #obj.crappyIO('fields'+str(n).rjust(2,'0'))
            if((self.stats.coupling_energy < 0.01**self.obj.dim*self.domain_measure*self.obj.Young) and (n > 5)):
                break
            self.stats.iteration_reset()
            self.obj.F_tot += self.strain_step_tensor*self.strain_step_scalar
            n += 1
           
