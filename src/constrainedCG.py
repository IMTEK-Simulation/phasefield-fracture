from mpi4py import MPI
import numpy as np
import time
## Code uses the algorithm discussed in Vollebregt (2014) (BCCG) with the Polak-Ribiere
## update formula in Eq. 16.  It enforces the constraint x >= field_constraint

class constrainedCG():

    def __init__(self, x_in,A,b,field_constraint, comm, maxiter = 20000, cg_tol = 1e-10, verbose=False):
        self.name = "constrainedCG"
        start = time.time()
        
        x = np.copy(x_in)
        mask_neg = x <= field_constraint    # Vollebregt step 2 
        x[mask_neg] = np.copy(field_constraint[mask_neg])    # Vollebregt step 0
    
        # check for early termination: entire set is constrained and feasible
        if (comm.allreduce(np.all(mask_neg),MPI.LAND) and comm.allreduce(np.all(b <= 0),MPI.LAND)):
            if ((comm.rank == 0) and (verbose)):
                print('Entire set is constrained and feasible.  Terminating early.')
            self.time = time.time() - start
            self.result = x
            self.n_iterations = 0
            self.residual = 0.0
            return
   
        r = A(x) - b
        mask_res = r >= 0
        mask_na = np.logical_and(mask_neg,mask_res)     
        r[mask_na] = 0.0      # Vollebregt step 3
        p = -r
    
        # check for early termination: residual meets tolerance
        rmax = comm.allreduce(np.max(abs(r)),MPI.MAX)
        if (rmax <= cg_tol):
            if ((comm.rank == 0) and (verbose)):
                print('Residual meets tolerance before iteration.  Terminating early.')
            self.time = time.time() - start
            self.result = x
            self.n_iterations = 0
            self.residual = rmax
            return
    
        for i in range (1, maxiter+1):
            Ap = A(p)
            denominator_temp =  comm.allreduce((p*Ap).sum(),MPI.SUM)
            alpha = - comm.allreduce((r * p).sum(),MPI.SUM) / denominator_temp    # Vollebregt step 4, modified per Eq. 16
            x += alpha*p
            mask_neg = x <= field_constraint   # Vollebregt step 5
            x[mask_neg] = np.copy(field_constraint[mask_neg])
            assert np.logical_not(np.isnan(x).any())
            r_old = r   # Vollebregt step 6
            r = A(x) - b
            mask_res = r > 0        # Vollebregt step 3a, again
            mask_bounded = np.logical_and(mask_neg,mask_res)
            r[mask_bounded] = 0.0
            rmax = comm.allreduce(np.max(abs(r)),MPI.MAX)
            if ((comm.rank == 0) and (verbose)):
                print('Iteration {}, max residual is {}'.format(i+1, rmax))
            if(rmax <= cg_tol):
                if ((comm.rank == 0) and (verbose)):
                    print('CG converged after {} iterations with max residual {}'.format(i+1, rmax))
                self.time = time.time() - start
                self.result = x
                self.n_iterations = i+1
                self.residual = rmax
                return
            beta =  comm.allreduce((r*(r - r_old)).sum(),MPI.SUM) / (alpha*denominator_temp) # Vollebregt step 3, modified per Eq. 16
            p_old = p
            p = -r + beta * p_old
            p[np.logical_not(mask_bounded)] = -r[np.logical_not(mask_bounded)] + beta*p_old[np.logical_not(mask_bounded)]
            p[mask_bounded] = 0.0
        
        if (comm.rank == 0):  
            print('CG did not converge after {} iterations.  Current max residual is {}'.format(i+1, rmax)) 
        self.time = time.time() - start
        self.result = x
        self.n_iterations = i+1
        self.residual = rmax
        return
