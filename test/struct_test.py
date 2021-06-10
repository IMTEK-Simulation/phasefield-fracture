import sys
sys.path.append("/Users/andrews/code/quasistatic-parallel-2D/src/")
import makestruct_flex
import numpy as np

nx=255
Lx=100
lamb = 4
lamb_width = 2
sigma = 0.3
minval = 1e-2

np.random.seed()
field = np.random.normal(0.0,1.0,size=(nx,nx))
field = makestruct_flex.wavelength_filter2D(field, (lamb-lamb_width/2)/Lx, sigma)
field = makestruct_flex.wavelength_filter2D(field, (lamb+lamb_width/2)/Lx, sigma, hipass=True)
field += 1
field = makestruct_flex.smoothcutoff2D(field, minval)
np.save('teststruct.npy',field)

propdict = {}
for variable in ["nx", "Lx", "lamb", "lamb_width", "sigma", "minval"]:
    propdict[variable] = eval(variable)
makestruct_flex.save_params(propdict)


