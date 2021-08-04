import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def get_data(dir, keynames, maxind, output_dup = False):
    data = {}
    for key in keynames:
        data[key] = []
    n = 0
    f = open(Path(dir)/'stats.json')
    for line in f:
        obj = json.loads(line)
        if (output_dup == True):
            if (obj["output_flag"] == True):
                obj = json.loads(line)
        for key in keynames:
            data[key].append(obj[key])
        n += 1
        if (n >= maxind):
            break
    for key in keynames:
        data[key] = np.array(data[key])
    return data

def get_data_output(dir, keynames, maxind, subiteration=False):
    f = open(Path(dir)/'stats.json')
    data = {}
    for key in keynames:
        data[key] = []
    m = 0
    for line in f:
        obj = json.loads(line)
        if (subiteration == False):
            if ((obj["output_flag"] == True)):
                for key in keynames:
                    data[key].append(obj[key])
                m += 1
        else:
            if ((obj["subit_output_flag"] == True)):
                for key in keynames:
                    data[key].append(obj[key])
                m += 1
        if (m >= maxind):
            break
    for key in keynames:
        data[key] = np.array(data[key])
    return data
