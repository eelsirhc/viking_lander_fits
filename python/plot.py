#!/usr/bin/env python
import asciitable
import numpy
import pylab
from fit_core import *
from argh import ArghParser, arg, command
from ordereddict import OrderedDict as odict
import itertools
def colors():
#13cols    mycol = ["#e00000", "#00c000", "#2040af", "#146ebe", "#0096fa", "#00c8fa",
#                 "#00b4be", "#a0e064", "#c8e040", "#fae040", "#eb9620", "#506ed2"]
    mycol = ["black","#e00000","#00c000","#146ebe","#7a2182","#217582","#d0cd1c"]
    return itertools.cycle(mycol)
    
def plotrange(new, old=None):
    if old is None:
        old=[9e36,-9e36]
    new[0] = new[0] if new[0] < old[0] else old[0]
    new[1] = new[1] if new[1] > old[1] else old[1]
    return new
    
def segmented_plot(x,y,color,ylim=None,label=None):
    diff = numpy.where(x[1:]-x[:-1] < 0)
    if len(diff[0]) == 0:
        pylab.plot(x,y,label=label, color=color)
    else:
        low =list(1.0+diff[0])
        high=list(diff[0])
        
        low.insert(0,0)
        high.append(-1)
        count=0
        for l,h in zip(low, high):
            if count==0:
                pylab.plot(x[l:h],y[l:h],color=color, label=label)
            else:
                pylab.plot(x[l:h],y[l:h],color=color)
            count=count+1
    ylim = plotrange([y.min(),y.max()], ylim)
    return ylim
    
def plot_dict(values, figsize=(6,4), filename="plot.png", dpi=150):
    """Iterates through the values dictionary, plotting each one"""
    lc = colors()
    pylab.figure(figsize=figsize)
    ylim=None
    for key, val in values.items():
        ylim=segmented_plot(val["x"],val["y"],label=key,ylim=ylim,color=next(lc))
    box = pylab.gca().get_position()
    nlines = int(len(values.keys())/3.)
    pylab.gca().set_position([box.x0,box.y0, box.width, box.height*(1.0-nlines*0.015)])
    leg = pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=3, mode="expand", borderaxespad=0., prop={'size':6})
    leg.get_frame().set_alpha(0.4)
    pylab.xlabel("Ls") ; pylab.xlim(0,360) ; pylab.xticks([0,90,180,270,360])
    pylab.ylabel("Pressure (Pa)") ; pylab.ylim(ylim)
    pylab.savefig(filename=filename, dpi=dpi)

def read_data(fname, lander, startrow=0, stoprow=-1):
    parameter="p1"
    if lander=="vl2":
        parameter="p2"

    if fname.endswith(".data") :
        indata = asciitable.read(fname)
        data = dict(x=indata["L_S"][startrow:stoprow],
                        y=indata[lander][startrow:stoprow])
    elif fname.endswith(".fitted"):
        indata = asciitable.read(fname)
        data = dict(x=indata["L_S"],
                        y=indata[lander])
    elif fname.endswith(".fit"):
        indata = asciitable.read(fname)
        ls = numpy.arange(360)
        data = dict(x=ls, y=fitfunc(indata[parameter],ls))
    else:
        raise AttributeError("Unknown Type")
    return data

@arg("filenames", nargs="+", type=str)
@arg("--output", default="plot.png", type=str)
@arg("--lander", default="vl1", type=str)
@arg("--startrow", default=1, type=int)
@arg("--stoprow", default=-1, type=int)
@arg("--baseline", default=None, type=str)
def plot_data(args):
    filenames = args.filenames
    values=odict()
    #for fit functions

    for fname in filenames:
        print fname
        try:
            values[fname]=read_data(fname, args.lander, args.startrow, args.stoprow)
        except AttributeError as e:
            pass
    
    if args.baseline is not None:
        base_data = read_data(fname, args.lander, args.startrow, args.stoprow)
        for key, val in values.items():
            if len(val["y"])!=len(base_data["y"]):
                print len(val["y"]), len(base_data["y"])
                raise ValueError("Need matching arrays to calculate perturbation")
            val["y"]=val["y"]-base_data["y"]
    plot_dict(values, filename=args.output)

@arg("filenames", nargs="+", type=str)
@arg("--output", default="plot.png", type=str)
@arg("--lander", default="vl1", type=str)
def plot_fit(args):
    filenames = args.filenames
    values=odict()
    parameter="p1"
    if args.lander=="vl1":
        parameter="p2"
    for fname in filenames:
        data = asciitable.read(fname)
        ls = numpy.arange(360)
        values[fname] = dict(x=ls, y=fitfunc(data[parameter],ls))
    plot_dict(values, filename=args.output)


if __name__=="__main__":
    parser = ArghParser()
    parser.add_commands([plot_data, plot_fit])
    parser.dispatch()