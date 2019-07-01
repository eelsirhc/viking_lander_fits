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
        fit.to_csv(output_filename)

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
    fit = core.fit_data(data, nmodes)
    # output the file
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    #
    fit.to_csv(output_filename)

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

    fit = core.fit_data(data, nmodes)
    # output the file
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    fit = fit.rename(columns=dict(Pressure=lander.lower()))
    
    fit.to_csv(output_filename)

@fit.command()
@click.argument("parameter_filename",type=str)
@click.argument("viking_filename", type=str)
@click.option("--suffix", type=str, default="low")
def jacobian(parameter_filename, viking_filename, suffix="low"):
	parameters = pd.read_csv(parameter_filename, comment="#",index_col=0)
	viking = core.read_file(viking_filename)
	
	for simulation in parameters.index:
		print(simulation)
		if simulation.endswith(suffix):
			data = core.read_file("fit/{}.fit".format(simulation))
			print(data.columns)

	#core.read_file(pname)
	viking_sc = core.convert_AP_to_SC(viking)

if __name__ == "__main__":
    cli()
