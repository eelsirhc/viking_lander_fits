#!/usr/bin/env python
#Christopher Lee, Ashima Research, 2013
import numpy
import argparse
from scipy import optimize, linalg
from fit_core import *


        
    result=dict()
    result["names"] = [data[n]["deltax_name"] for n in data]
    result["basevalue"] = [base[data[n]["deltax_name"]] for n in data]
    result["perturbation"] = [x for x in X[0]]
    result["newvalue"] = [a+b for a,b in zip(result["basevalue"],result["perturbation"])]

    target = open(args.output_filename_parameters, 'w')
    target.write("#Parameter fit to best reproduce {0} data\n".format(args.lander_name))
    
#   Fit to VL data from {0} with {1} harmonic modes\n".format(args.input_filename, args.nmodes))
    asciitable.write(result,target, delimiter=args.delimiter, 
                    names=["names","basevalue","perturbation","newvalue"], 
                    formats=dict(perturbation="%.4f", newvalue="%.4f"))
    
    target.close()
    
    #print base["data"].keys()
    result2 = dict(L_S=base["data"]["L_S"], 
                    vl1=base["data"]["fit_vl1"], 
                    vl2=base["data"]["fit_vl2"],
                    res_vl1=base["data"]["res_vl1"], 
                    res_vl2=base["data"]["res_vl2"],
                    )
    
    target2 = open(args.output_filename_fit, 'w')
    target2.write("#Best fit pressure curve to {0} data\n".format(args.lander_name))
    for line in open(args.output_filename_parameters):
        target2.write("#"+line)
    asciitable.write(result2,target2, delimiter=args.delimiter, 
                    names=["L_S","vl1","vl2", "res_vl1","res_vl2"])
    
    target2.close()
