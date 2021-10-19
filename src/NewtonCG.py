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

from mpi4py import MPI
import numpy as np


##################### Newton-CG algorithm ####################################
def parallelNewtonCG(x0,jacobian,hessp, comm, maxiter_newton = 100, newton_tol = 1e-8):
    # equation to solve iteratively: J(x) = 0
    # linear system: He(x_n)delta = - J(x_n)
    # update: x_{n+1} = x_n + delta
    x = x0
    for iternum_newton in range(maxiter_newton):
        Jx = -jacobian(x)
        residual = np.sqrt(comm.allreduce((Jx*Jx).sum(),MPI.SUM))
        if(comm.rank == 0):
            print('Newton residual at step {} is {}'.format(iternum_newton+1,residual))
        if (residual < newton_tol):
            break
        delta = parallelCG(Jx,x,hessp, Jx, comm)
        deltanorm = comm.allreduce((delta*delta).sum(),MPI.SUM)
        if(comm.rank == 0):
            print('norm of update is {}'.format(deltanorm))
        x = x + delta
      #  print(x)
    return x

def parallelCG(delta,x,A,b, comm, maxiter_cg = 10000, cg_tol = 1e-8):
    # CG algorithm, copied shamelessly from the CG wikipedia page and parallelized
    # (delta is now the x from the wiki code, since x from the Newton iteration is
    # needed for the hessian)
    r = b - A(x,delta)
    p = r
    rsold = comm.allreduce((r*r).sum(),MPI.SUM)
    for iternum_cg in range(maxiter_cg):
        Ap = A(x,p)
        pAp = comm.allreduce((p*Ap).sum(),MPI.SUM)
        alpha = rsold/pAp
        delta = delta + alpha*p
        r = r - alpha*Ap
        rsnew = comm.allreduce((r*r).sum(),MPI.SUM)
        if (np.sqrt(rsold) < cg_tol):
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
   #     if (comm.rank == 0):
   #         print(rsold)
    if (comm.rank == 0):
        if (iternum_cg+1 < maxiter_cg):
            print('CG converged after {} iterations with residual {}'.format(iternum_cg+1, rsnew))
        else:
            print('CG reached max iterations without converging; returing anyway')
    return delta