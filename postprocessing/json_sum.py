import json
import numpy as np
from pathlib import Path

def json_sum(rundir, verbose=False):
    if (verbose == True):
        print(rundir)
    f = open(Path(rundir)/'stats_fullit.json')
    totaldict = {}
    for line in f:
        obj = json.loads(line)
        for key in obj:
            if key in totaldict:
                if(( key == 'elastic_CG_its') or (key == 'elastic_newton_its')):
                    totaldict[key] = obj[key]
                else:
                    totaldict[key] += obj[key]
            else:
                totaldict[key] = obj[key]
        if 'solver time' in totaldict:
            totaldict['solver time'] += obj['elastic_time'] + obj['phi_time']
        else:
            totaldict['solver time'] = obj['elastic_time'] + obj['phi_time']
    f.close()
    if (verbose == True):
        for key in totaldict:
            print('total '+key+': '+str(totaldict[key]))
    return totaldict
        
class json_analyze():

    def __init__(self, rundirs=None, rootdir='.'):
        self.rootdir = rootdir
        if (rundirs==None):
            rundirs = []
            p = Path(rootdir)
            for x in p.iterdir():
                if ((x/'stats_fullit.json').exists()):
                    rundirs.append(str(x))
                if (x.is_dir()):
                    for y in x.iterdir():
                        if ((y/'stats_fullit.json').exists()):
                            rundirs.append(str(y))
            self.rundirs = sorted(rundirs)
        else:
            self.rundirs = rundirs
        print(self.rundirs)
        self.get_superdict()
            
    def get_superdict(self):
        self.superdict = {}
        for dirname in self.rundirs:
            self.superdict[dirname] = json_sum(dirname)
            
    def query_attribute(self, attr):
        for dirname in self.rundirs:
            if attr in self.superdict[dirname]:
                print('For run in {0: <40}'.format(dirname+',')+'total '+attr+' is: '+str(self.superdict[dirname][attr]))
            else:
                print('No attribute '+attr+' in output for '+dirname)

