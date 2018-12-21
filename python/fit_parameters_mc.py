#!/usr/bin/env python
#Christopher Lee, Ashima Research, 2013
from __future__ import print_function
import numpy
import argparse
from scipy import optimize, linalg
from fit_core import *
import itertools
import curses

def f(args):
    index,a=args
    if len(a) ==3:
        A,b,weights=a
    elif len(a) == 6:
        A,b,weights, bv, lv, hv=a
    numpy.random.seed(index)
    x=numpy.random.randn(A.shape[1])*0.1
    ro = rms(errfunc2(A.T,x,b,weights))

    for i in range(1000):
        #choose an entry at random
        elem = int(numpy.random.random(1)*len(x))
        #perturb
        xo=x[elem]
        x[elem]=numpy.random.randn(1)*0.1
        if elem>2:
            x[elem]=x[elem]*10.
        
        #test
        rn = rms(errfunc2(A.T,x,b,weights))
        if rn > ro or any((bv+x) < lv) or any((bv+x) > hv) or numpy.random.random(1) > 0.95:
            #larger, reset
            #or randomly reset to get out of local minima
            x[elem]=xo
        else:
            ro=rn
    print(index)
    return x,ro
    
def rms(x):
    return numpy.sqrt(numpy.mean(x**2))
        
def errfunc2(p, x, y, w):
    """Error function defined as the difference between f(x) and data"""
    return w*(fitfunc2(p,x)-y)

def fitfunc2(p, x):
    """generates sum(p[i]*x[i])"""
    val=0.0*x[0]
    for n in range(len(p)):
        val=val+p[n]*x[n]
    return val

def find_delta_x(data, basename):
    base = data[basename]
    dn = base.keys()
    excluded = ["filename", "data"]
    parameter_names = list(set(dn).difference(set(excluded)))
    
    for k,v in data.items():
        #skip if this entry is the baseline entry
        if k==basename:
            continue
        #iterate over parameters to find the delta
        delta = dict([(pn, v[pn]-base[pn]) for pn in parameter_names])
        #how many are non zero
        nonzero=0
        for pn,vn in delta.items():
           if abs(vn) > 1e-10:
                nonzero+=1
                result=vn
                result_name=pn
        if nonzero>1:
            print("Error, found too many non-zeroes perturbations")
        v["deltax"]=result
        v["deltax_name"]=result_name
        #now delta p
        v["data"]["delta_p_vl1"] = v["data"]["vl1"] - base["data"]["vl1"]
        v["data"]["delta_p_vl2"] = v["data"]["vl2"] - base["data"]["vl2"]
        v["data"]["delta_f_vl1"] = v["data"]["p1"]  - base["data"]["p1"]
        v["data"]["delta_f_vl2"] = v["data"]["p2"]  - base["data"]["p1"]
        v["base_value"] = base[result_name]
    return data,parameter_names

def fit_parameters(parameter_file, viking, lander="vl1", delimiter=",", start_from=None):
    #load the data from the fit files
    data, basename = load_files(parameter_file, delimiter=delimiter)
    #load the viking data
    viking_data = read_file(viking, delimiter=delimiter)
    
    ls = numpy.arange(360.)
    #regenerate 4360 data points from the harmonic fits in 'data'
    for k,v in data.items():
        d=v["data"]
        #regenerate the L_S dependent data
        d["L_S"] = ls
        d["vl1"] = fitfunc(d["p1"], d["L_S"])
        d["vl2"] = fitfunc(d["p2"], d["L_S"])

    #generate the smoothed viking data
    viking_data["L_S"] = ls
    viking_data["vl1"] = fitfunc(viking_data["p1"], viking_data["L_S"])
    viking_data["vl2"] = fitfunc(viking_data["p2"], viking_data["L_S"])

    #calculate perturbations
    dp = []
    dx = []
    dp2= []
    #calculate the delta_x for each row, and the delta pressure from the base state
    find_delta_x(data, basename)
    
    #remove the base state, it's not needed for the fit
    base = data.pop(basename)
    #calculate the difference between viking and base.
    if start_from is not None:
        infile = read_file(start_from, delimiter=delimiter)
        base["data"]["L_S"] = ls
        base["data"]["vl1"] = fitfunc(infile["p1"], d["L_S"])
        base["data"]["vl2"] = fitfunc(infile["p2"], d["L_S"])
    delta = {}
    for l in ["vl1","vl2"]:
        delta[l] = viking_data[l] - base["data"][l]
    delta["both"] = numpy.hstack((delta["vl1"],delta["vl2"]))
    name=[]
    #loop through data, extracting the state vector dX and the perturbations dP for vl1,vl2
    base_values = []
    min_values = []
    max_values = []
    for key, val in data.items():
        base_values.append(val["base_value"])
        if val["deltax_name"].count("albedo"):
            min_values.append(0)
            max_values.append(1)
        elif val["deltax_name"].count("ice_cond"):
            min_values.append(1.)
            max_values.append(8.)
        else:
            min_values.append(0.)
            max_values.append(1e9)

        
        dp.append( val["data"]["delta_p_vl1".format(lander)] )
        dp2.append( val["data"]["delta_p_vl2".format(lander)] )
        dx.append( val["deltax"] )
        name.append(key)
    base_values, min_values, max_values = numpy.array(base_values), numpy.array(min_values), numpy.array(max_values)
    #create the A matrix (Ax=B) from the dP,dX
    a1=numpy.array([p/x for p,x in zip(dp, dx)]).T
    a2=numpy.array([p/x for p,x in zip(dp2, dx)]).T
    
    #store the a matrices
    a=dict(vl1=a1, vl2=a2, both=numpy.vstack((a1,a2)))
    #start with the initial guess of the perturbation as 0.1 everywhere
    p0 = numpy.zeros(a[lander].shape[1], dtype=numpy.float64)+0.1
    #if we want to weight things unevenly we can weight it with the gauss function
    def gauss(x, m, s):
        delta = numpy.abs(x-m)
        delta[delta > 180] = 360.-delta[delta > 180]

        return numpy.exp(-(delta/s)**2)
    #constant weights for now
    if lander=="both":
        ls=numpy.hstack((ls,ls))
    weights = 1.0 + gauss(ls,120,40)*2.0 - gauss(ls,330,40)*0.5
    #call optimize with errfunc2 to calculate Ax'-P to allow leastsq to minimize it to find x.
    p1, success =  optimize.leastsq(errfunc2, 
                    p0[:], 
                    args=(a[lander].T, delta[lander], weights))
    #have out state vector
    X=[p1]
    #X is the perturbation state vector, so we calculate p + A.X to calculate our estimated p_new
    vl1 = numpy.zeros(len(base["data"]["L_S"])) + base["data"]["vl1"]
    vl2 = numpy.zeros(len(base["data"]["L_S"])) + base["data"]["vl2"]
    
    for s, p1,p2 in zip(X[0], a["vl1"].T,a["vl2"].T ): 
        vl1 = vl1 + s * p1
        vl2 = vl2 + s * p2
    #store the estimated pressure curve, and the residuals.
    base["data"]["fit_vl1"]=vl1
    base["data"]["fit_vl2"]=vl2
    base["data"]["res_vl1"]=viking_data["vl1"]-vl1
    base["data"]["res_vl2"]=viking_data["vl2"]-vl2


    #try and MCMC
    #a1 is the A array, whose rows are the deltas attributed to the change in parameter x
    A=a["vl1"]
    #delta[lander] is the b array whose values we are trying to match
    b=delta[lander]
    
    #initial
    #loop
    import pylab as pl
    pl.figure()
    

    from multiprocessing import Pool
    p=Pool(8)
#    for args in itertools.izip_longest(range(1000),[(A,b,weights)],fillvalue=(A,b,weights)):
    res = p.map(f,itertools.izip_longest(range(10000),[(A,b,weights,base_values, min_values, max_values)],fillvalue=(A,b,weights, base_values, min_values, max_values)))
    m=[]
    rr=[]

    for x,r in res:
        m.append(x)
        rr.append(r)
        vl0 = numpy.zeros(len(base["data"]["L_S"])) + base["data"]["vl1"]
        for s, p1 in zip(x, A.T):
             vl0 =vl0 + s * p1
        pl.plot(vl0,label="mc", alpha=0.005,color='k')
    
    pl.plot(vl1,label="fit")
    pl.plot(viking_data["vl1"],label="vl")
#    pl.legend()
    pl.savefig("a.png")
    #end MCMC
    m=numpy.vstack(m)
    q=numpy.vstack([m.T,numpy.array(rr)])
    numpy.savetxt("mdata",q.T,delimiter=',',fmt="%8.5f")
    print(numpy.mean(m,axis=0))
    print(numpy.std(m,axis=0))
    return data, base, X
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parameter_file", type=str)
    parser.add_argument("viking", type=str)
    parser.add_argument("output_filename_parameters", type=str)
    parser.add_argument("output_filename_fit", type=str)
    parser.add_argument("--delimiter", type=str, default=',')
    parser.add_argument("--lander_name", type=str, default="vl1")
    parser.add_argument("--monte_carlo", "-m", type=int, default=0)
    parser.add_argument("--start_from", "-s", type=str, default=None)

    args = parser.parse_args()
    
    if args.monte_carlo == 0:
        data,base,X = fit_parameters(args.parameter_file, args.viking,lander=args.lander_name, 
                                    delimiter=args.delimiter, start_from=args.start_from)
        print("RMS={0}".format(numpy.sqrt(numpy.mean(base["data"]["res_vl1"]**2))))
    else:
        #monte carlo fit
        input_data = asciitable.read(args.parameter_file)
        lendata = len(input_data)
        results=dict()
        import progressbar
        for i in progressbar.ProgressBar()(range(args.monte_carlo)):
            pick = int(numpy.random.rand(1)*(lendata-1)+1)
            newdata = numpy.hstack((input_data[0:1],numpy.random.choice(input_data[1:],pick)))
            asciitable.write(newdata, open("tempfile",'w'), delimiter=args.delimiter)
            data,base,X = fit_parameters("tempfile", 
                            args.viking,lander=args.lander_name, delimiter=args.delimiter)
            for k,v in zip(data.keys(), X[0]):
                r=results.get(k,[])
                r.append([v,numpy.std(base["data"]["res_vl1"])])
                results[k]=r
        for k in sorted(results.keys()):
            r=results[k]
            data, weights = [numpy.array(q) for q in zip(*r)]
            _x_,x_x = numpy.average(data, weights=weights),numpy.average(data**2, weights=weights)
            m,s = _x_, numpy.sqrt(x_x-_x_**2)
            print("{0} = {1} +- {2}".format(k,m,s))
            numpy.savetxt("{0}.data".format(k), zip(data,weights))
        import sys
        sys.exit(0)
        
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
