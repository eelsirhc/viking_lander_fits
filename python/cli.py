#!/usr/bin/env python
# Christopher Lee, Aeolis Research, 2019

import click
import numpy as np
import pandas as pd
import core
import os


@click.group()
def cli():
    pass


@cli.group()
def fit():
    pass


@fit.command()
@click.argument("filename")
@click.option("--startrow", default=0, type=int)
@click.option("--stoprow", default=None, type=int)
@click.option("--delimiter", default=",", type=str)
@click.option("--nmodes", default=5, type=int)
def models(
    filename, startrow=0, stoprow=None, delimiter=None, nmodes=5
):
    # read the file into memory
    runfile = pd.read_csv(filename,comment="#")
    for simulation in runfile.name:
        print(simulation)
	    # fit the file
        infilename = "data/{}.data".format(simulation)
        data = core.read_file(
            infilename, startrow=startrow, stoprow=stoprow, delimiter=delimiter
    	    )
        fit = core.fit_data(data, nmodes)
        # output the file
        output_filename = "fit/{}.fit".format(simulation)
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))
        fit.to_csv(output_filename, index_label="modes")

@fit.command()
@click.argument("filename")
@click.argument("output_filename")
@click.option("--startrow", default=0, type=int)
@click.option("--stoprow", default=None, type=int)
@click.option("--delimiter", default=",", type=str)
@click.option("--nmodes", default=5, type=int)
def model(
    filename, output_filename, startrow=0, stoprow=None, delimiter=None, nmodes=5
):
    # read the file into memory
    data = core.read_file(
        filename, startrow=startrow, stoprow=stoprow, delimiter=delimiter
    )
    # fit the file
    fitted = core.fit_data(data, nmodes)
    # output the file
    directory = os.path.dirname(output_filename)
    if not directory and directory!="":
        os.makedirs(directory)
    #
    fitted.to_csv(output_filename, index_label="modes")

def csvi(s):
	return [int(i) for i in s.split(",")]

@fit.command()
@click.argument("filename")
@click.argument("output_filename")
@click.option("--lander", default="VL1", type=str)
@click.option("--years", default=None, type=csvi)
@click.option("--nmodes", default=5, type=int)
def viking(filename, output_filename, lander, years, nmodes=5):
    
    if years is None:
    	years = dict(VL1=[2,3],VL2=[2])[lander]

    data = core.read_viking(filename, lander, years)

    fitted = core.fit_data(data, nmodes)
    # output the file
    directory = os.path.dirname(output_filename)
    if not directory and directory!="":
        os.makedirs(directory)
    fitted = fitted.rename(columns=dict(Pressure=lander.lower()))
    
    fitted.to_csv(output_filename, index_label="modes")

@fit.command()
@click.argument("parameter_filename",type=str)
@click.argument("viking_filename", type=str)
@click.option("--suffix", type=str, default="low")
def parameters(parameter_filename, viking_filename, suffix="low"):
	parameters = pd.read_csv(parameter_filename, comment="#",index_col=0)
	viking = core.convert_AP_to_SC(
				core.read_file(viking_filename)
				)
	data = []
	for simulation in parameters.index:
		if simulation.endswith(suffix) or\
		   simulation=="base":
			df = core.convert_AP_to_SC(
					core.read_file("fit/{}.fit".format(simulation))
					)[viking.columns]
			cols = dict(zip(viking.columns,viking.columns))
			for k in cols.keys():
				if k not in ["L_S","modes"]:
					cols[k]=simulation
			
			del cols["modes"]
			df = df.set_index("modes")			
			data.append(df.rename(columns=cols))

	data = pd.concat(data,axis=1)
	dy = data.copy()
	for c in [x for x in data.columns if x != "base"]:
		dy[c]-=dy["base"]
	del dy["base"]

	dx = parameters.T
	for c in [x for x in dx.columns if x != "base"]:
		dx[c]=(dx[c]-dx["base"])/dx["base"]
	dx = dx[dy.columns].T
	#print(dx)

	
	#print(viking)
	#print(data)
	delta = viking.set_index("modes")["vl1"]-data["base"]
	#print(delta)
	popt, pstd, dydx = core.least_square_inversion(dy,dx,delta)
	
	import matplotlib.pyplot as plt
	plt.figure(figsize=(12,8))
	ls = np.arange(360)
	for d in data:
		print(d)
		print(data)
		print(pd.DataFrame(data[d]))

		yy = core.calc_AP_SC(ls,pd.DataFrame(data[d]))

		plt.plot(ls,yy, label=d)
	plt.savefig("a.png")
	print( data["base"])
	print( data["base"].shape, dydx.values.shape, popt.shape)
	print( data["base"]+np.dot(dydx.values,popt))
	#ls = np.arange(360)
	#core.calc_AP_SC(ls,res[0])

from plotting import register_plots
register_plots(cli)

if __name__ == "__main__":
    cli()
