#
# Copyright 2021 W. Beck Andrews
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import mechanics
import model_components
import parallel_fracture
import numpy as np
import json
import time
import statlog


class simulation():
    
    def __init__(self, obj, time_dependent=False):
        self.obj = obj
        self.nmax = 1000
        self.mmax = 5
        self.output_n = 6
        self.subit_outputname = 'altmin.nc'
        self.fullit_outputname = 'test.nc'
        self.statsname = 'stats.json'
        self.timingname = 'timing.json'
        self.paramname = 'simulation.json'
        
     #   self.delta_energy_tol = 1e-6*obj.domain_measure
        self.delta_phi_tol = 1e-3
        self.dt0 = 2**16
        self.dtmin = 2**(-16)
        self.dphidt_lim = 1.5
        self.overforce_lim = 0.75
        self.stiffness_end = 100.0
        self.rescaling_flag = False

        self.time_dependent = time_dependent
        if (time_dependent == False):
            self.dt0 = 0.0
            self.dtmin = 0.0
            self.dphidt_lim = 0.0
            self.overforce_lim = 0.0
        
        self.stats = statlog.stats(self.statsname, obj)
        self.timing = statlog.timing(self.timingname)
        self.strain_step_tensor = np.array([[0,0],[0,1.0]])
        self.strain_step_scalar = 0.0002
        self.min_its = 10

    def subiteration(self):
        start = time.time() 
        strain_solve = self.obj.strain_solver()
        straint = time.time()
        self.obj.mechform.get_elastic_coupling(self.obj, strain_solve)
        self.stats.get_stats_stress(strain_solve.stress,self.obj)
        self.stats.get_stats_elastic(self.obj)
        if(self.time_dependent == False):
            phi_solve = self.obj.phi_solver()
        else:
            if ((self.stats.max_overforce > self.overforce_lim) and
                    (self.rescaling_flag == False) and 
                    (self.stats.coupling_at_ofmax > 1.0)):
                self.rescaling_flag = True
            if (self.rescaling_flag == True):
                ratio = self.rescaleF()
                self.stats.get_stats_stress(strain_solve.stress*ratio,self.obj)
            phi_solve = self.obj.phi_implicit_solver()
        self.obj.phi.array()[...] += phi_solve.result
        self.stats.get_stats_phi(phi_solve.result, self.obj)
        self.stats.get_stats_full(self.obj)
        phit = time.time()
        # timing stats
        self.timing.elastic_CG_its = strain_solve.nb_fev
        self.timing.elastic_newton_its = strain_solve.nb_it
        self.timing.elastic_time = straint-start
        self.timing.phi_subits = phi_solve.n_iterations
        self.timing.phi_time = phit-straint
        self.timing.iteration_update()

    def rescaleF(self):
        phi_only = (self.stats.max_overforce - self.stats.coupling_at_ofmax)
        coupling_target = (self.overforce_lim - phi_only)
        ratio = (coupling_target/self.stats.coupling_at_ofmax)**0.5
        ratio = min(ratio, 1 + self.strain_step_scalar/np.max(np.abs(self.obj.F_tot)))
        self.obj.straineng.array()[...] *= ratio**2
        self.obj.strain.array()[...] *= ratio
        self.obj.F_tot *= ratio
        self.stats.get_stats_rescale(self.obj, ratio)
        return ratio

    def altmin_iteration(self):
        self.obj.F_tot += self.strain_step_tensor*self.strain_step_scalar
        self.stats.subiteration = 0
        self.timing.subiteration = 0
        self.stats.delta_phi = 1e8
        while(abs(self.stats.delta_phi) > self.delta_phi_tol):
            self.subiteration()
            if(self.obj.comm.rank == 0):
                statlog.dump(self.timing, self.timingname)
                print('delta energy = ', self.stats.delta_energy,
                    'delta phi = ',self.stats.delta_phi)
            if((self.stats.subiteration > 1) and ((self.stats.subiteration % self.output_n == 0)
                    or (self.stats.subiteration % self.output_n == 1)
                    or (abs(self.stats.delta_energy) > 0.02*self.stats.total_energy))):
                if(self.obj.comm.rank == 0):
                    print('saving subiteration # ', self.stats.subiteration,
                     'as output number', self.stats.subit_output_index)
                    self.stats.subit_output_dump()
                self.obj.muOutput(self.subit_outputname)
            else:
                if(self.obj.comm.rank == 0):
                    statlog.dump(self.stats, self.statsname)
        if(self.obj.comm.rank == 0):
            print('strain: ', self.stats.strain, 'energy: ',
                self.stats.total_energy,'delta phi',self.stats.delta_phi)
            self.stats.output_dump()
        self.obj.muOutput(self.fullit_outputname)                 

    def timedep_iteration(self):
        self.obj.F_tot += self.strain_step_tensor*self.strain_step_scalar
        self.stats.subiteration = 0
        self.timing.subiteration = 0
        self.obj.dt = self.dt0
        while True:
            while True:
                self.subiteration()
                if((self.stats.delta_phi > self.dphidt_lim) and (self.obj.dt > self.dtmin)):
                    self.obj.dt /= 2
                    if(self.obj.comm.rank == 0):
                        print('decreasing timestep, delta phi = ', self.stats.delta_phi)
                    self.obj.phi.array()[...] = self.obj.phi_old + 0.0
                else:
                    if ((self.stats.delta_phi < self.dphidt_lim/2)
                            and (self.obj.dt < self.dt0)):
                        self.obj.dt *= 2
                    break
            if (self.obj.comm.rank == 0):
                statlog.dump(self.timing, self.timingname)
                print('overforce max, ', self.stats.max_overforce, ', dt = ', self.obj.dt,
                    'fracture energy', self.stats.total_fracture_energy)
                print('energy', self.stats.total_energy, 'delta energy = ', self.stats.delta_energy,
                   'delta phi = ', self.stats.delta_phi)
            self.obj.phi_old = self.obj.phi.array() + 0.0
            if ((abs(self.stats.delta_phi) < self.delta_phi_tol) and 
                     (self.obj.dt >= self.dt0 )):
                if(self.overforce_lim > 1e6):
                    #  non-overforce-limited case might need new strain increment
                    if(self.obj.comm.rank == 0):
                        self.stats.output_dump()
                        print('saving implicit timestep # ', self.stats.subiteration,
                            ' with energy = ', self.stats.total_energy,
                            'avg strain = ', self.stats.strain)
                    self.obj.muOutput(self.fullit_outputname)
                elif (self.stats.stress/self.stats.strain < self.stiffness_end):
                    # only output once for end of simulation
                    if(self.obj.comm.rank == 0):
                        self.stats.output_dump()
                        print('saving implicit timestep # ', self.stats.subiteration,
                            ' with energy = ', self.stats.total_energy,
                            'avg strain = ', self.stats.strain)
                    self.obj.muOutput(self.fullit_outputname)
                break
            if((self.stats.subiteration > 1) and 
                    ((self.stats.subiteration % self.output_n == 0) or
                    (self.stats.subiteration % self.output_n == 1) or 
                    (abs(self.stats.delta_energy) > 0.04*self.stats.total_energy))):
                if(self.obj.comm.rank == 0):
                    self.stats.output_dump()
                    print('saving implicit timestep # ', self.stats.subiteration,
                        ' with energy = ', self.stats.total_energy,
                        'avg strain = ', self.stats.strain)
                self.obj.muOutput(self.fullit_outputname)
            else:
                if(self.obj.comm.rank == 0):
                    statlog.dump(self.stats, self.stats.fname)

    def run_simulation(self):
        self.obj.phi_old = self.obj.phi.array() + 0.0
        phi_solve = self.obj.phi_solver()
        self.obj.phi.array()[...] += phi_solve.result
        self.obj.phi_old = self.obj.phi.array() + 0.0
        self.obj.F_tot = self.strain_step_scalar*self.strain_step_tensor
        self.obj.muOutput(self.fullit_outputname,new=True)
        if (self.time_dependent == False):
            self.obj.muOutput(self.subit_outputname,new=True)
        if(self.obj.comm.rank == 0):
            statlog.clear(self.statsname)
            self.stats.output_dump()
            statlog.clear(self.paramname)
            statlog.dump(self,self.paramname)
        n = 0
        while ((n < self.nmax)):
            self.obj.phi_old = np.maximum(self.obj.phi.array(), self.obj.phi_old)
            if (self.time_dependent == False):
                self.altmin_iteration()
            else:
                self.timedep_iteration()
            if(self.obj.comm.rank == 0):
                print("effective stiffness :", self.stats.stress/self.stats.strain)
            if (self.stats.stress/self.stats.strain < self.stiffness_end):
                break
            n += 1

        if(self.obj.comm.rank == 0):
            statlog.dump(self,self.paramname)

