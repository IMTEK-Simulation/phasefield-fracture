import numpy as np
from mpi4py import MPI

import sys
mu_build_path = "/home/fr/fr_fr/fr_wa1005/muspectre_stuff/builds/muspectre-20210202/build/"
sys.path.append(mu_build_path + '/language_bindings/python')
sys.path.append(mu_build_path + '/language_bindings/libmufft/python')
sys.path.append(mu_build_path + '/language_bindings/libmugrid/python')

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
            pixel_id = np.ravel_multi_index(pixel, self.fftengine.nb_subdomain_grid_pts, order='F')
            material.add_pixel(pixel_id, obj.Cx.array()[tuple(pixel)]*interp[tuple(pixel)], obj.Poisson)
        return material

    def update_material(self, obj):            
        ### set current material properties
        interp = obj.interp.energy(obj.phi.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, self.fftengine.nb_subdomain_grid_pts, order='C')
            self.material.set_youngs_modulus(pixel_id,Cxval*interp[tuple(pixel)])

    def get_elastic_coupling(self, obj, strain_result):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(strain_result.grad[pixel_id*4:(pixel_id+1)*4],(2,2))
            obj.strain.array()[:,0,pixel[0],pixel[1]] = strain.flatten()
            trace = 0.0
            for k in range(0,self.dim):
                trace += strain[k,k]
            obj.straineng.array()[tuple(pixel)] = 0.5*obj.Cx.array()[tuple(pixel)]*(2.0*mu_factor*(strain**2).sum() 
                + lamb_factor*obj.Cx.array()[tuple(pixel)]*trace**2)
                
    def get_elastic_energy(self, obj):
        return obj.straineng.array()*obj.interp.energy(obj.phi.array())

class anisotropic_tc():
    def __init__(self,obj):
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
                lamb_factor*0.5 + np.sum(np.maximum(pstrains,0))**2*mu_factor)
        
    def get_compressive_energy(self,obj):
        lamb_factor = obj.Poisson/(1+obj.Poisson)/(1-2*obj.Poisson)
        mu_factor = 1/2/(1+obj.Poisson)
        compressive_energy = np.zeros_like(obj.straineng.array())
        for pixel, Cxval in np.ndenumerate(obj.Cx.array()):
            pixel_id = np.ravel_multi_index(pixel, obj.fftengine.nb_subdomain_grid_pts, order='F')
            strain = np.reshape(obj.strain.array()[:,0,pixel[0],pixel[1]],(obj.dim,obj.dim))
            pstrains = np.linalg.eigvalsh(strain)
            compressive_energy[tuple(pixel)] = Cxval*(np.minimum(np.sum(pstrains),0)**2*lamb_factor*0.5 +
                np.sum(np.minimum(pstrains,0))**2*mu_factor)
        return compressive_energy
        
    def get_elastic_energy(self, obj):
        return (obj.interp.energy(obj.phi.array())*obj.straineng.array() + self.get_compressive_energy(obj))
