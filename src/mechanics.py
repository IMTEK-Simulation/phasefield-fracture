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

import numpy as np
from mpi4py import MPI

import sys
import muSpectre as msp
import muFFT
import muGrid

class isotropic_tc():
    def __init__(self):
        self.name = "isotropic_tc"
            
    def initialize_material(self,obj):
        material = msp.material.MaterialLinearElastic4_2d.make(obj.cell, "material_small")
        interp = obj.interp.energy(obj.phi.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            material.add_pixel(pixel_id, obj.Cx.array()[tuple(pixel)]*interp[tuple(pixel)], obj.Poisson)
        return material

    def update_material(self, obj):            
        ### set current material properties
        interp = obj.interp.energy(obj.phi.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='C')
            obj.material.set_youngs_modulus(pixel_id,Cxval*interp[tuple(pixel)])

    def get_elastic_coupling(self, obj, strain_result):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(strain_result.grad[pixel_id*obj.dim**2:(pixel_id+1)*obj.dim**2],(obj.dim,obj.dim))
            obj.strain.array()[:,0,pixel[0],pixel[1]] = strain.flatten()
            trace = 0.0
            for k in range(0,obj.dim):
                trace += strain[k,k]
            obj.straineng.array()[tuple(pixel)] = 0.5*obj.Cx.array()[tuple(pixel)]*(2.0*mu_factor*(strain**2).sum() 
                + lamb_factor*trace**2)
                
    def get_elastic_energy(self, obj):
        return obj.straineng.array()*obj.interp.energy(obj.phi.array())

class anisotropic_tc():
    def __init__(self):
        self.name = "anisotropic_tc"

    def initialize_material(self,obj):
        material = msp.material.MaterialPhaseFieldFracture_2d.make(obj.cell, "material_small",obj.ksmall)
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
             pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
             material.add_pixel(pixel_id, obj.Cx.array()[tuple(pixel)], 
                 obj.Poisson, obj.phi.array()[tuple(pixel)])
        return material

    def update_material(self,obj):            
        ### set current material properties
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='C')
            obj.material.set_phase_field(pixel_id, obj.phi.array()[tuple(pixel)])
    
### key parts of test system
    def get_elastic_coupling(self,obj,strain_result):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(strain_result.grad[pixel_id*obj.dim**2:(pixel_id+1)*obj.dim**2],(obj.dim,obj.dim))
            obj.strain.array()[:,0,pixel[0],pixel[1]] = strain.flatten()
            pstrains = np.linalg.eigvalsh(strain)
            obj.straineng.array()[tuple(pixel)] = Cxval*(np.maximum(np.sum(pstrains),0)**2*
                lamb_factor*0.5 + np.sum(np.maximum(pstrains,0)**2)*mu_factor)
        
    def get_compressive_energy(self,obj):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        compressive_energy = np.zeros_like(obj.straineng.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(obj.strain.array()[:,0,pixel[0],pixel[1]],(obj.dim,obj.dim))
            pstrains = np.linalg.eigvalsh(strain)
            compressive_energy[tuple(pixel)] = Cxval*(np.minimum(np.sum(pstrains),0)**2*lamb_factor*0.5 +
                np.sum(np.minimum(pstrains,0)**2)*mu_factor)
        return compressive_energy
        
    def get_elastic_energy(self, obj):
        return (obj.interp.energy(obj.phi.array())*obj.straineng.array() + self.get_compressive_energy(obj))

# copy of isotropic_tc that shuts off driving force for compression
class nonvar_phi_tc():
    def __init__(self):
        self.name = "nonvar_phi_tc"
            
    def initialize_material(self,obj):
        material = msp.material.MaterialLinearElastic4_2d.make(obj.cell, "material_small")
        interp = obj.interp.energy(obj.phi.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            material.add_pixel(pixel_id, obj.Cx.array()[tuple(pixel)]*interp[tuple(pixel)], obj.Poisson)
        return material

    def update_material(self, obj):            
        ### set current material properties
        interp = obj.interp.energy(obj.phi.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='C')
            obj.material.set_youngs_modulus(pixel_id,Cxval*interp[tuple(pixel)])

    def get_elastic_coupling(self,obj,strain_result):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(strain_result.grad[pixel_id*obj.dim**2:(pixel_id+1)*obj.dim**2],(obj.dim,obj.dim))
            obj.strain.array()[:,0,pixel[0],pixel[1]] = strain.flatten()
            pstrains = np.linalg.eigvalsh(strain)
            obj.straineng.array()[tuple(pixel)] = Cxval*(np.maximum(np.sum(pstrains),0)**2*
                lamb_factor*0.5 + np.sum(np.maximum(pstrains,0)**2)*mu_factor)

    def get_elastic_energy(self, obj):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        elastic_energy = np.zeros_like(obj.straineng.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(obj.strain.array()[:,0,pixel[0],pixel[1]],(obj.dim,obj.dim))            
            obj.strain.array()[:,0,pixel[0],pixel[1]] = strain.flatten()
            trace = 0.0
            for k in range(0,obj.dim):
                trace += strain[k,k]
            elastic_energy[tuple(pixel)] = 0.5*obj.Cx.array()[tuple(pixel)]*(2.0*mu_factor*(strain**2).sum()
                + lamb_factor*trace**2)

# copy of anisotropic_tc that uses isotropic elastic energy to evolve phi
class nonvar_contact_tc():
    def __init__(self):
        self.name = "nonvar_phi_tc"
            
    def initialize_material(self,obj):
        material = msp.material.MaterialPhaseFieldFracture_2d.make(obj.cell, "material_small",obj.ksmall)
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
             pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
             material.add_pixel(pixel_id, obj.Cx.array()[tuple(pixel)], 
                 obj.Poisson, obj.phi.array()[tuple(pixel)])
        return material

    def update_material(self,obj):            
        ### set current material properties
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='C')
            obj.material.set_phase_field(pixel_id, obj.phi.array()[tuple(pixel)])

    def get_elastic_coupling(self, obj, strain_result):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(strain_result.grad[pixel_id*obj.dim**2:(pixel_id+1)*obj.dim**2],(obj.dim,obj.dim))
            obj.strain.array()[:,0,pixel[0],pixel[1]] = strain.flatten()
            trace = 0.0
            for k in range(0,obj.dim):
                trace += strain[k,k]
            obj.straineng.array()[tuple(pixel)] = 0.5*obj.Cx.array()[tuple(pixel)]*(2.0*mu_factor*(strain**2).sum()
                + lamb_factor*trace**2)

    def get_elastic_energy(self,obj):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        elastic_energy = np.zeros_like(obj.straineng.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(obj.strain.array()[:,0,pixel[0],pixel[1]],(obj.dim,obj.dim))
            pstrains = np.linalg.eigvalsh(strain)
            elastic_energy[tuple(pixel)] += Cxval*(np.minimum(np.sum(pstrains),0)**2*lamb_factor*0.5 +
                np.sum(np.minimum(pstrains,0)**2)*mu_factor)
            elastic_energy[tuple(pixel)] += Cxval*(np.maximum(np.sum(pstrains),0)**2*lamb_factor*0.5 +
                np.sum(np.maximum(pstrains,0)**2)*mu_factor)*obj.interp.energy(obj.phi.array()[pixel[0],pixel[1]])
        return elastic_energy


class voldev_tc():
    def __init__(self):
        self.name = "voldev_tc"

    def initialize_material(self,obj):
        material = msp.material.MaterialPhaseFieldFracture2_2d.make(obj.cell, "material_small",obj.ksmall)
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
             pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
             material.add_pixel(pixel_id, obj.Cx.array()[tuple(pixel)], 
                 obj.Poisson, obj.phi.array()[tuple(pixel)])
        return material

    def update_material(self,obj):            
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='C')
            obj.material.set_phase_field(pixel_id, obj.phi.array()[tuple(pixel)])
    
    def get_elastic_coupling(self,obj,strain_result):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        K_factor = lamb_factor + 2/3*mu_factor
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(strain_result.grad[pixel_id*obj.dim**2:(pixel_id+1)*obj.dim**2],(obj.dim,obj.dim))
            obj.strain.array()[:,0,pixel[0],pixel[1]] = strain.flatten()
            trace = 0.0
            for k in range(0,obj.dim):
                trace += strain[k,k]
            obj.straineng.array()[tuple(pixel)] = Cxval*(np.maximum(trace,0)**2*
                K_factor*0.5 + np.sum((strain-np.eye(obj.dim)*trace/3)**2)*mu_factor)
        
    def get_compressive_energy(self,obj):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        K_factor = lamb_factor + 2/3*mu_factor
        compressive_energy = np.zeros_like(obj.straineng.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(obj.strain.array()[:,0,pixel[0],pixel[1]],(obj.dim,obj.dim))
            trace = 0.0
            for k in range(0,obj.dim):
                trace += strain[k,k]
            compressive_energy[tuple(pixel)] = Cxval*(np.minimum(trace,0)**2*K_factor*0.5)
        return compressive_energy
        
    def get_elastic_energy(self, obj):
        return (obj.interp.energy(obj.phi.array())*obj.straineng.array() + self.get_compressive_energy(obj))

# non-variational model with volumetric/deviatoric split for contact and spectral split for phase field
class nonvar_both_tc():
    def __init__(self):
        self.name = "nonvar_both_tc"

    def initialize_material(self,obj):
        material = msp.material.MaterialPhaseFieldFracture2_2d.make(obj.cell, "material_small",obj.ksmall)
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
             pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
             material.add_pixel(pixel_id, obj.Cx.array()[tuple(pixel)],
                 obj.Poisson, obj.phi.array()[tuple(pixel)])
        return material

    def update_material(self,obj):
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='C')
            obj.material.set_phase_field(pixel_id, obj.phi.array()[tuple(pixel)])

    def get_elastic_coupling(self,obj,strain_result):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(strain_result.grad[pixel_id*obj.dim**2:(pixel_id+1)*obj.dim**2],(obj.dim,obj.dim))
            obj.strain.array()[:,0,pixel[0],pixel[1]] = strain.flatten()
            pstrains = np.linalg.eigvalsh(strain)
            obj.straineng.array()[tuple(pixel)] = Cxval*(np.maximum(np.sum(pstrains),0)**2*
                lamb_factor*0.5 + np.sum(np.maximum(pstrains,0)**2)*mu_factor)

    def get_elastic_energy(self,obj):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        K_factor = lamb_factor + 2/3*mu_factor
        elastic_energy = np.zeros_like(obj.straineng.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(obj.strain.array()[:,0,pixel[0],pixel[1]],(obj.dim,obj.dim))
            trace = 0.0
            for k in range(0,obj.dim):
                trace += strain[k,k]
            elastic_energy[tuple(pixel)] += (Cxval*(np.maximum(trace,0)**2*
                K_factor*0.5 + np.sum((strain-np.eye(obj.dim)*trace/3)**2)*
                mu_factor))*obj.interp.energy(obj.phi.array()[pixel[0],pixel[1]])
            elastic_energy[tuple(pixel)] += Cxval*(np.minimum(trace,0)**2*K_factor*0.5)
        return elastic_energy

class nonvar_phi_vd():
    def __init__(self):
        self.name = "nonvar_phi_vd"

    def initialize_material(self,obj):
        material = msp.material.MaterialLinearElastic4_2d.make(obj.cell, "material_small")
        interp = obj.interp.energy(obj.phi.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            material.add_pixel(pixel_id, obj.Cx.array()[tuple(pixel)]*interp[tuple(pixel)], obj.Poisson)
        return material

    def update_material(self, obj):
        ### set current material properties
        interp = obj.interp.energy(obj.phi.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='C')
            obj.material.set_youngs_modulus(pixel_id,Cxval*interp[tuple(pixel)])

    def get_elastic_coupling(self,obj,strain_result):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        K_factor = lamb_factor + 2/3*mu_factor
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(strain_result.grad[pixel_id*obj.dim**2:(pixel_id+1)*obj.dim**2],(obj.dim,obj.dim))
            obj.strain.array()[:,0,pixel[0],pixel[1]] = strain.flatten()
            trace = 0.0
            for k in range(0,obj.dim):
                trace += strain[k,k]
            obj.straineng.array()[tuple(pixel)] = Cxval*(np.maximum(trace,0)**2*
                K_factor*0.5 + np.sum((strain-np.eye(obj.dim)*trace/3)**2)*mu_factor)
        
    def get_elastic_energy(self,obj):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        K_factor = lamb_factor + 2/3*mu_factor
        elastic_energy = np.zeros_like(obj.straineng.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(obj.strain.array()[:,0,pixel[0],pixel[1]],(obj.dim,obj.dim))
            trace = 0.0
            for k in range(0,obj.dim):
                trace += strain[k,k]
            elastic_energy[tuple(pixel)] += (Cxval*(np.maximum(trace,0)**2*
                K_factor*0.5 + np.sum((strain-np.eye(obj.dim)*trace/3)**2)*
                mu_factor))*obj.interp.energy(obj.phi.array()[pixel[0],pixel[1]])
            elastic_energy[tuple(pixel)] += Cxval*(np.minimum(trace,0)**2*K_factor*0.5)
        return elastic_energy
