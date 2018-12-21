#!/usr/bin/env python
#Christopher Lee, Ashima Research, 2013
from __future__ import print_function
import numpy
import asciitable
import argparse
import os
from fit_core import *
import asciitable


def fit_file(filename, startrow=0, stoprow=None, delimiter=None,nmodes=5):
    data =read_file(filename, 
                    startrow=startrow, stoprow=stoprow, 
                    delimiter=delimiter)
    #fit VL1 data
    print(data)
    fit= fit_data(data, nmodes)
    return fit

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", type=str)
    parser.add_argument("output_filename", type=str)
    parser.add_argument("--startrow", type=int, default=0)
    parser.add_argument("--stoprow", type=int, default=None)
    parser.add_argument("--delimiter", type=str, default=',')
    parser.add_argument("--nmodes", type=int, default=5)

    args = parser.parse_args()
    
    fit = fit_file(args.input_filename, 
                    startrow=args.startrow, 
                    stoprow=args.stoprow, 
                    delimiter=args.delimiter, 
                    nmodes=args.nmodes)

    output = dict(mode=range(len(fit["p1"])), 
                  p1=fit["p1"], 
                  p2=fit["p2"])
    target = open(args.output_filename, 'w')
    target.write("#Fit to VL data from {0} with {1} harmonic modes\n".format(args.input_filename, args.nmodes))
    asciitable.write(output,target, delimiter=args.delimiter)
    target.close()
