#!/usr/bin/env python
import asciitable
import numpy
import pylab
from fit_core import *
from argh import ArghParser, arg
from collections import OrderedDict as odict
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
    
def segmented_plot(x,y,color,ylim=None,label=None,ls=None,lw=None, return_handle=False,alpha=0.5):
    diff = numpy.where(x[1:]-x[:-1] < 0)
    lw = lw or 1
    ls = ls or "-"
    if len(diff[0]) == 0:
        handle = pylab.plot(x,y,label=label, color=color, alpha=alpha,ls=ls,lw=lw)
    else:
        low =list(1.0+diff[0])
        high=list(diff[0])
        
        low.insert(0,0)
        high.append(-1)
        count=0
        for l,h in zip(low, high):
            if count==0:
                handle = pylab.plot(x[l:h],y[l:h],color=color, label=label, alpha=alpha,ls=ls,lw=lw)
            else:
                pylab.plot(x[l:h],y[l:h],color=color, alpha=alpha,ls=ls,lw=lw)
            count=count+1
    ylim = plotrange([y.min(),y.max()], ylim)
    if return_handle:
        return ylim, handle
    else:
        return ylim
    
def plot_dict(values, figsize=(6,4), filename="plot.png", dpi=150):
    """Iterates through the values dictionary, plotting each one"""
    lc = colors()
    pylab.figure(figsize=figsize)
    ylim=None
    for key, val in values.items():

        ylim=segmented_plot(val["x"]%360,val["y"],label=key,ylim=ylim,color=next(lc))
    box = pylab.gca().get_position()
    nlines = int(len(values.keys())/3.)
    pylab.gca().set_position([box.x0,box.y0, box.width, box.height*(1.0-nlines*0.015)])
    leg = pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=3, mode="expand", borderaxespad=0., prop={'size':6})
    leg.get_frame().set_alpha(0.4)
    pylab.xlabel("Ls") ; pylab.xlim(0,360) ; pylab.xticks([0,90,180,270,360])
    pylab.ylabel("Pressure (Pa)") ; pylab.ylim(ylim)
    pylab.savefig(filename=filename, dpi=dpi)

def rms_dict(values, filename="rms.dat"):
    """Iterates through the values dictionary in pairs calculating RMS"""
    def rms_pair(a,b):
        return numpy.mean(numpy.sqrt((a["y"]-b["y"])**2))

    columns=["name"]
    formats=dict(name=lambda x: x)
    rows=[]
    for key2 in values.keys():
        columns.append(key2)
        formats[key2]="%4.1F"
    
    for key1 in values.keys():
        r=[key1]
        for key2 in values.keys():
            r.append(rms_pair(values[key1], values[key2]))
        rows.append(r)
    asciitable.write(rows, names=columns, formats=formats, delimiter=",")

def plot_2dict(values, values2, figsize=(6,4), filename="plot.png", dpi=150):
    """Iterates through the values dictionary, plotting each one"""
    pylab.figure(figsize=figsize)
    for count,valdict in enumerate([values,values2]):
        ylim=None
        lc = colors()
        pylab.subplot(2,1,count+1)
        for key, val in valdict.items():
            ylim=segmented_plot(val["x"],val["y"],label=key,ylim=ylim,color=next(lc))
        box = pylab.gca().get_position()
        nlines = int(len(values.keys())/3.)
        pylab.gca().set_position([box.x0,box.y0, box.width, box.height*(1.0-nlines*0.015)])
        leg = pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=3, mode="expand", borderaxespad=0., prop={'size':6})
        leg.get_frame().set_alpha(0.4)
        pylab.xlabel("Ls") ; pylab.xlim(0,360) ; pylab.xticks([0,90,180,270,360])
        pylab.ylabel("Pressure (Pa)") ; pylab.ylim(ylim)
    #
    pylab.savefig(filename=filename, dpi=dpi)

def read_data(fname, lander, startrow=0, stoprow=-1):
    parameter="p1"
    if lander=="vl2":
        parameter="p2"

    if fname.endswith(".data") :
        indata = read_file(fname, startrow, stoprow, delimiter=",")
        data = dict(x=indata["L_S"],
                        y=indata[lander])
    elif fname.endswith(".fitted"):
        indata = asciitable.read(fname)
        data = dict(x=indata["L_S"],
                        y=indata[lander])
    elif fname.endswith(".fit"):
        indata = asciitable.read(fname)
        ls = numpy.arange(360)
        data = dict(x=ls, y=fitfunc(indata[parameter],ls), parameter = indata[parameter])
    else:
        raise AttributeError("Unknown Type")
    return data

@arg("filenames", nargs="+", type=str)
@arg("--output", default="plot.png", type=str)
@arg("--lander", default="vl1", type=str)
@arg("--startrow", default=1, type=int)
@arg("--stoprow", default=-1, type=int)
@arg("--baseline", default=None, type=str)
def plot_data(filenames, output="plot.png",lander="vl1",startrow=1,stoprow=-1,baseline=None):
    values=odict()
    #for fit functions

    for fname in filenames:
        print fname
        try:
            values[fname]=read_data(fname, lander, startrow, stoprow)
        except AttributeError as e:
            pass
    
    if baseline is not None:
        base_data = read_data(baseline, lander, startrow, stoprow)
        for key, val in values.items():
            if len(val["y"])!=len(base_data["y"]):
                print len(val["y"]), len(base_data["y"])
                raise ValueError("Need matching arrays to calculate perturbation")
            val["y"]=val["y"]-base_data["y"]
            print key, numpy.mean(val["y"]), numpy.std(val["y"])
    plot_dict(values, filename=output)


@arg("viking", type=str)
@arg("data", type=str)
@arg("fit", type=str)
@arg("--output", default="plot.png", type=str)
@arg("--startrow", default=1, type=int)
@arg("--stoprow", default=-1, type=int)
@arg("--baseline", default=None, type=str)
def plot_xin(viking,data,fit,output="plot.png",startrow=1,stoprow=-1,baseline=None):
    values=odict()
    #for fit functions
    
    values["vl1"] = odict()
    values["vl2"] = odict()
    
    for name in ["viking","data","fit"]:
        fname = getattr(args,name)
        try:
            values["vl1"][name]=read_data(fname, "vl1", args.startrow, args.stoprow)
            values["vl2"][name]=read_data(fname, "vl2", args.startrow, args.stoprow)
        except AttributeError as e:
            pass
    
    for v in ["vl1","vl2"]:
        values[v]["diff"] = dict(values[v]["viking"])
        values[v]["diff"]["y"] = values[v]["diff"]["y"] - values[v]["fit"]["y"]
    

    pylab.figure(figsize=(6,6))
    ylim=None
    
    plots = odict()
    top_plots = [  ["vl1",values["vl1"]["data"], "grey","-",1],
                    [None,values["vl1"]["fit"], "magenta","--",2],    
                    [None,values["vl1"]["viking"], "blue","--",2],
                  ["vl2",values["vl2"]["data"], "black","-",1],
                    [None,values["vl2"]["fit"], "green","--",2],    
                    [None,values["vl2"]["viking"], "red","--",2]]

    bottom_plots = [ ["vl1",values["vl1"]["diff"], "blue","-",2],
                     ["vl2",values["vl2"]["diff"], "red","-",2] ]

    ax = pylab.subplot2grid((9,1),(0, 0),rowspan=6)
    handles = odict()
    for n,p,c,l,w in top_plots:
        ylim,handle=segmented_plot(p["x"],p["y"],ylim=ylim, color=c, ls=l,lw=w, return_handle=True,alpha=1)
        if n:
            handles[n] = handle
    pylab.legend(handles,loc='upper left', framealpha=0.5)
    pylab.xlabel("Ls") ; pylab.xlim(0,360) ; pylab.xticks([0,60,120,180,240,300,360])
    pylab.ylabel("Pressure (Pa)") ; pylab.ylim(ylim)
    pylab.grid()
    
    
    ax = pylab.subplot2grid((9,1),(7, 0),rowspan=2)
    ylim=None
    handles = odict()
    for n,p,c,l,w in bottom_plots:
        ylim,handle=segmented_plot(p["x"],p["y"],ylim=ylim, color=c, ls=l,lw=w, return_handle=True,alpha=1)
        if n:
            handles[n] = handle
    pylab.legend(handles, loc="upper left", framealpha=0.5)
    pylab.xlabel("Ls") ; pylab.xlim(0,360) ; pylab.xticks([0,60,120,180,240,300,360])
    pylab.ylabel("P (Pa)") ; pylab.ylim(ylim)
    pylab.grid()
    
    pylab.savefig(filename=args.output, dpi=150)


@arg("data", type=str)
@arg("fit", type=str)
@arg("--output", default="plot.png", type=str)
@arg("--lander", default="vl1", type=str)
@arg("--startrow", default=1, type=int)
@arg("--stoprow", default=-1, type=int)
def plot_fit(data, fit, output="plot.png",lander="vl1",startrow=1,stoprow=-1):

    data = read_data(data, lander, startrow, stoprow)
    fit =  read_data(fit, lander)
    
    fit["x"] = data["x"]
    fit["y"] = fitfunc(fit["parameter"],fit["x"])
    
    values = odict()
    values["data"]=data
    values["fit"]=fit

    values2 = odict()
#    values2["data"]=data
    values2["residual"]=dict(fit)
    values2["residual"]["y"] = data["y"] - fit["y"]
    print numpy.mean(values2["residual"]["y"]), numpy.std(values2["residual"]["y"]),\
        numpy.sqrt(numpy.mean(values2["residual"]["y"]**2))
    
    plot_2dict(values, values2, filename=output)

@arg("filename", type=str)
@arg("output", type=str)
@arg("--dpi", default=150, type=int)
@arg("--figsize", default=None, type=lambda x: [int(y) for y in x.split(",")])
def plot_fitted(filename,output, dpi=150,figsize=None):
    figsize=figsize or (6,4)
    data = asciitable.read(args.filename)
    pylab.figure(figsize=args.figsize)
    pylab.subplots_adjust(hspace=0.25)
    ax1 = pylab.subplot2grid((3,1),(0,0),rowspan=2)
    ax1.plot(data["L_S"], data["vl1"]+data["res_vl1"], color='blue',label="VL1")
    ax1.plot(data["L_S"], data["vl2"]+data["res_vl2"], color='red',label="VL2")
    ax1.plot(data["L_S"], data["vl1"], color='magenta',ls="--", label="fit vl1")
    ax1.plot(data["L_S"], data["vl2"], color='green', ls="--", label="fit vl2")
    ax1.set_xlabel("Ls") ; ax1.set_xlim(0,360) ; ax1.set_xticks([0,90,180,270,360])
    ax1.set_ylabel("Pressure (Pa)")
    ax1.legend(framealpha=0.2,prop={'size':5})

    ax2 = pylab.subplot2grid((3,1),(2,0),rowspan=1)
    ax2.plot(data["L_S"], data["res_vl1"], color='blue', label="res vl1")
    ax2.plot(data["L_S"], data["res_vl2"], color='red', label="res vl2")
    ax2.legend(framealpha=0.2,loc='lower right',prop={'size':5})
    ax2.set_xlabel("Ls") ; ax2.set_xlim(0,360) ; ax2.set_xticks([0,90,180,270,360])
    ax2.set_ylabel("Pressure (Pa)")
    ax2.set_yticks(ax2.get_yticks()[::2])
    pylab.savefig(filename=args.output, dpi=args.dpi)

@arg("filenames", nargs="+", type=str)
@arg("--output", default="rms.dat", type=str)
@arg("--lander", default="vl1", type=str)
@arg("--startrow", default=1, type=int)
@arg("--stoprow", default=-1, type=int)
def rms_data(filenames,output="rms.dat",lander="vl1",startrow=1,stoprow=-1):
    filenames = args.filenames
    values=odict()
    #for fit functions

    for fname in filenames:
        print fname
        try:
            values[fname]=read_data(fname, args.lander, args.startrow, args.stoprow)
        except AttributeError as e:
            pass
    
    rms_dict(values, filename=args.output)

if __name__=="__main__":
    parser = ArghParser()
    parser.add_commands([plot_data, plot_fit, plot_fitted, rms_data, plot_xin])
    parser.dispatch()
