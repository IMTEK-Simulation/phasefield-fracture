import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def get_data(dir, keynames, maxind, output_dup = False):
    f = open(Path(dir)/'stats.json')
    data = {}
    for key in keynames:
        data[key] = np.zeros(maxind)
    n = 0
    for line in f:
        obj = json.loads(line)
        if (output_dup == True):
            if (obj["output_flag"] == True):
                obj = json.loads(line)
        for key in keynames:
            data[key][n] = obj[key]
        n += 1
        if (n >= maxind):
            break
        
    return data

def get_data_output(dir, keynames, outinds, subiteration=False):
    f = open(Path(dir)/'stats.json')
    data = {}
    for key in keynames:
        data[key] = np.zeros(outinds.size)
    m = 0
    for line in f:
        obj = json.loads(line)
        if (subiteraton == False):
            if ((obj["output_index"] == outinds[m]) and (obj["output_flag"] == True)):
                for key in keynames:
                    data[key] = obj[key]
                m += 1
        else:
            if ((obj["subit_output_index"] == outinds[m]) and (obj["subit_output_flag"] == True)):
                for key in keynames:
                    data[key] = obj[key]
                m += 1
        if (m >= outinds.size):
            break
    return data