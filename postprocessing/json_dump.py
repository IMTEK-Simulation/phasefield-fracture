import json
import numpy as np
rootdir = 'big_scaling1/c09_1/'
fname = 'stats.json'
f = open(rootdir+fname)
strain_time = 0
phi_time = 0
subits = 0
for line in f:
    obj = json.loads(line)
    print(obj['avg_strain'],obj['strain_time'],obj['phi_time'])
    strain_time += obj['strain_time']
    phi_time += obj['phi_time']
    subits += obj['subiterations']
    

print(strain_time,phi_time,subits)
f.close()
