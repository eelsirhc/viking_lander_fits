from scipy import optimize, linalg
import numpy
import asciitable
from ordereddict import OrderedDict as odict

def load_files(filename_list, delimiter=","):
    """Reads a filename+parameter list"""
    header=asciitable.read(filename_list)
    d_data = odict()
    for line in header:
        entry=dict()
        for name in line.dtype.names:
            entry[name]=line[name]
        entry["data"] = read_file(line["filename"],delimiter=delimiter)
                                    
        d_data[line["filename"]] = entry
        
    return d_data, header[0]["filename"]

def read_file(filename, startrow = 0, stoprow=None,delimiter=None):
    """Reads the data file from the filename, optionally skipping number of rows,
    using an optional delimiter """
    pathname = filename
    data = asciitable.read(pathname, delimiter=delimiter, numpy=False)
    if stoprow is None:
        stoprow=None
    sl=slice(startrow, stoprow,None)
    for key in data.keys():
        data[key] = numpy.array(data[key][startrow:stoprow])
    return data

def errfunc(p, x, y):
    """Error function defined as the difference between f(x) and data"""
    return (fitfunc(p,x)-y)

def fitfunc(p, x):
    """Model function, defined as y = mean + sum( amp[i]*cos( 2*pi*mode*(t-phase[i])/360. ) )"""
    val = p[0]
    for n in range(0,len(p)-1,2):
        ind = n+1
        mode = (n/2)+1
        val = val + p[ind]*numpy.cos(2*numpy.pi*mode*(x-p[ind+1])/360.)
    return val

def fit_data(data, nmodes):

    npars = 1+2*nmodes
    p0 = numpy.zeros(npars, dtype=numpy.float64)
    p1=None ; vl1=None
    p2=None ; vl2=None
    L_S = numpy.arange(360)
    fit_results=dict(L_S=L_S)
    if "vl1" in data:
        p1, success =  optimize.leastsq(errfunc, 
                        p0[:], 
                        args=(data["L_S"][1:-1].astype(numpy.float64), 
                              data["vl1"][1:-1].astype(numpy.float64)))
        vl1 = fitfunc(p1, L_S)
        fit_results["p1"]=p1
        fit_results["vl1"]=vl1
    if "vl2" in data:
        p2, success =  optimize.leastsq(errfunc, 
                        p0[:], 
                        args=(data["L_S"][1:-1].astype(numpy.float64), 
                              data["vl2"][1:-1].astype(numpy.float64)))
        vl2 = fitfunc(p2, L_S)
        fit_results["p2"]=p2
        fit_results["vl2"]=vl2

    return     fit_results
