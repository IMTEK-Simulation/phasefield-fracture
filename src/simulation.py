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
        self.nmax = 400
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
        self.delta_energy_tol = 1e-5
        self.total_energy = self.obj.objective(self.obj.phi.array())
        self.delta_energy = 0.0
        self.delta_phi = 0.0
        self.energy_old = 0.0
        self.strain_step_tensor = np.array([[0,0],[0,1.0]])
        self.strain_step_scalar = 0.0004
        self.min_strain_step = 0.000025
        self.min_its = 5
        self.dt0 = 2**20
        self.dtmin = 2**(-8)
        self.dphidt = 0.5
        self.couplinglim = 1.2

    def avg_strain(self):
        return np.max(np.linalg.eigvals(self.obj.F_tot))
        
    def subiteration(self, IsImplicit=False):
        subit_stats = statlog.altmin_iteration_stats(self.subit_statsname, self.avg_strain())
        self.energy_old = self.total_energy
        start = time.time() 
        strain_solve = self.obj.strain_solver()
        straint = time.time()
        self.obj.mechform.get_elastic_coupling(self.obj, strain_solve)
        if(IsImplicit==False):
            phi_solve = self.obj.phi_solver()
        else:
#            grad = self.obj.integrate(np.abs(self.obj.jacobian(self.obj.phi.array())))
#            step_radius = 10
#            self.obj.dt = min(step_radius/grad, step_radius)
            phi_solve = self.obj.phi_implicit_solver()
        self.obj.phi.array()[...] += phi_solve.result
        self.total_energy = self.obj.objective(self.obj.phi.array())
        phit = time.time()
        subit_stats.delta_energy = self.total_energy - self.energy_old
        subit_stats.delta_phi = self.obj.integrate(phi_solve.result)
#        print(subit_stats.delta_phi, self.obj.integrate(self.obj.phi.array()[...]-self.obj.phi_old))
        self.delta_energy = abs(subit_stats.delta_energy)
        self.delta_phi = abs(subit_stats.delta_phi)
        subit_stats.elastic_CG_its = strain_solve.nb_fev
        subit_stats.elastic_newton_its = strain_solve.nb_it
        subit_stats.elastic_time = straint-start
        subit_stats.phi_subits = phi_solve.n_iterations
        subit_stats.phi_time = phit-straint
        self.stats.subiteration_update(subit_stats)
        subit_stats.subiteration = self.stats.subiterations
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
                   and (self.stats.subiterations > 1)):
                if (self.strain_step_scalar <= self.min_strain_step):
                    self.obj.F_tot -= self.strain_step_tensor*self.strain_step_scalar
                    return
                else:
                    self.obj.F_tot -= self.strain_step_tensor*self.strain_step_scalar
                    self.strain_step_scalar /= 2
                    self.obj.F_tot += self.strain_step_tensor*self.strain_step_scalar
                    self.stats.subiterations = 1
                    delta_energy_old = 1e8
                    self.obj.phi.array()[...] = self.obj.phi_old + 0.0
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

    def implicit_iterator(self):
        delta_energy_timestep = 1e8
        self.obj.muOutput('timestep.nc',new=True)   
        if(self.obj.comm.rank == 0):
            print('beginning implicit timestepping')
        #while(delta_energy_timestep > self.delta_energy_tol*self.obj.dt):
        n = 0
        self.obj.dt = self.dt0
        while((self.stats.coupling_energy > 0.02*self.domain_measure) or
            (self.delta_energy > self.delta_energy_tol)):
       # while((self.delta_energy > self.delta_energy_tol*1e-2)):
            n = n+1
            energy_old = self.total_energy
            self.delta_phi = 0.0
            while True:
                self.subiteration(IsImplicit=True)
                if((self.delta_phi > self.dphidt) and (self.obj.dt > self.dtmin)):
                    self.obj.dt /= 2
                    if(self.obj.comm.rank == 0):
                        print('decreasing timestep, delta phi = ', self.delta_phi)
                    self.obj.phi.array()[...] = self.obj.phi_old + 0.0
                else:
                    if ((self.delta_phi < self.dphidt/2) and (self.obj.dt < self.dt0)):
                        self.obj.dt *= 2
                    break
            couplingmax = self.obj.max(self.obj.interp.energy(self.obj.phi_old)*self.obj.straineng.array())
            if (couplingmax > self.couplinglim):
                self.obj.F_tot *= (self.couplinglim/couplingmax)**0.5
            #self.obj.dt = 0.1
            if (self.obj.comm.rank == 0):
                print('couplingmax, ', couplingmax, ', dt = ', self.obj.dt)
                print('energy', self.total_energy, 'delta energy = ', self.delta_energy,
                   'delta phi = ', self.delta_phi)
            if (( self.delta_phi > 10.0) or (n % 10 == 0)):
                if(self.obj.comm.rank == 0):
                    print('saving implicit timestep # ', n,' with energy = ', self.total_energy)
                self.obj.muOutput('timestep.nc')
            self.obj.phi_old = self.obj.phi.array() + 0.0
            self.stats.coupling_energy = self.obj.integrate(self.obj.straineng*
                self.obj.interp.energy(self.obj.phi.array()))

    def run_simulation(self):
        self.obj.F_tot = self.strain_step_scalar*self.strain_step_tensor
        self.obj.phi_old = self.obj.phi.array() + 0.0
        self.obj.muOutput(self.fullit_outputname,new=True)
        self.stats = statlog.full_iteration_stats(self.fullit_statsname)
        n = 0
        while (n < self.nmax):
            if(self.strain_step_scalar > self.min_strain_step):
                self.iteration()
            else:
                self.implicit_iterator()
            self.stats.avg_strain = self.avg_strain()
            self.stats.total_energy = self.total_energy
            self.stats.delta_phi = self.obj.integrate(self.obj.phi.array()-self.obj.phi_old)
            self.stats.coupling_energy = self.obj.integrate(self.obj.straineng*
                self.obj.interp.energy(self.obj.phi.array()))

            self.obj.phi_old = self.obj.phi.array() + 0.0
            if(self.obj.comm.rank == 0):
                print('strain: ', self.stats.avg_strain, 'energy: ',
                    self.stats.total_energy,'delta phi',self.stats.delta_phi)
                self.stats.dump()
            self.obj.muOutput(self.fullit_outputname)
            #obj.crappyIO('fields'+str(n).rjust(2,'0'))
            if (((self.stats.coupling_energy < 0.02*self.domain_measure) or
                (self.stats.delta_phi > self.obj.lens[1]/2)) and (n > self.min_its)):
                break
            self.stats.iteration_reset()
            self.obj.F_tot += self.strain_step_tensor*self.strain_step_scalar
            n += 1
           
