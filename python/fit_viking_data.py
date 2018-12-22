#!/usr/bin/env python
#Christopher Lee, Ashima Research, 2013
#http://starbase.jpl.nasa.gov/vl1-m-met-4-binned-p-t-v-corr-v1.0/vl_1001/data/
#https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP/BZDSFA
from __future__ import print_function
import asciitable
import argparse
import sys
from netCDF4 import Dataset
from fit_core import *

def fit_viking_data(input_file, vl1years="2,3",  vl2years="2", nmodes=5, rems_too=False):
    """Translate viking data from
         name year ls  d h m s p(mb)
         VL1  1  97.073    0 18  0 49  7.534
         to the same format as the model data
         ls, vl1, vl2
    """
    
    data = asciitable.read(input_file)
    data.dtype.names=["name","year","L_S","day","hour","minute","second","pressure"]
    #scale to Pascal
    pressure =data["pressure"] *100.
    lander_name = data["name"]
    year = data["year"]
    L_S = data["L_S"]
    
    #vl1
    vl1_list_years = [int(y) for y in vl1years.split(",")]
    selection_year = [False]*len(year)
    for y in vl1_list_years:
        selection_year = selection_year | (year==y)
    selection = (lander_name=="VL1") & selection_year
    d_data = dict(L_S=L_S[selection],vl1=pressure[selection])
    fit= fit_data(d_data, nmodes)

    #vl2
    vl2_list_years = [int(y) for y in vl2years.split(",")]
    selection_year = [False]*len(year)
    for y in vl2_list_years:
        selection_year = selection_year | (year==y)    
    selection = (lander_name=="VL2") & selection_year
    d_data2 = dict(L_S=L_S[selection],vl2=pressure[selection])
    fit2= fit_data(d_data2, nmodes)
    
    fit.update(fit2)
#    d_data["vl2"] = d_data2["vl2"]
#    d_data["vl2_L_S"] = d_data2["L_S"]
    asciitable.write(d_data2, "vl2.data",delimiter=",")
    asciitable.write(d_data, "vl1.data",delimiter=",")
    return fit

def fit_rems_data(input_file, nmodes=5):
    """Read in Chris' netCDF version of the REMS data and fit it
    """

    try:
      nc = Dataset(input_file,'r',format='NETCDF3')
    except:
      print("You probably do not have the requiredrems_aggregated.nc")
      print("file located in the default location (or it's not in your")
      print("specified location. Specify it with --rems_filename and")
      print("grab it from:")
      print("https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP/BZDSFA")
      sys.exit()

    L_S = nc.variables["L_S"][:]
    pres= nc.variables["ADJUSTED_PRESSURE_AMBIENT_TEMP"][:]

    nc.close()

    d_data = dict(L_S=L_S,msl=pres)
    fit = fit_data(d_data, nmodes)
    asciitable.write(d_data, "msl.data",delimiter=",")
    return fit

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filename", type=str, default="vl_p.dat.txt")
    parser.add_argument("--output_filename", type=str, default="lander_data.fit")
    parser.add_argument("--vl1years", type=str, default="2,3")
    parser.add_argument("--vl2years", type=str, default="2")
#    parser.add_argument("--startrow", type=int, default=0)
#    parser.add_argument("--stoprow", type=int, default=None)
    parser.add_argument("--delimiter", type=str, default=',')
    parser.add_argument("--nmodes", type=int, default=5)
    parser.add_argument("--remstoo", action='store_true')
    parser.add_argument("--rems_filename", type=str, default="/Users/mir/REMS/Chris/rems_aggregated.nc")

    args = parser.parse_args()
    
    fit = fit_viking_data(args.input_filename, 
                    nmodes=args.nmodes,
                    vl1years = args.vl1years,
                    vl2years = args.vl2years
                    )

    if(args.remstoo):
        fit_rems = fit_rems_data(args.rems_filename,
                        nmodes=args.nmodes,
                        )
        fit.update(fit_rems)

    if(args.remstoo):
        output = dict(mode=range(len(fit["p1"])), 
                  p1=fit["p1"], 
                  p2=fit["p2"],
                  p3=fit["p3"])
    else:    
        output = dict(mode=range(len(fit["p1"])), 
                  p1=fit["p1"], 
                  p2=fit["p2"])
    target = open(args.output_filename, 'w')
    target.write("#Fit to VL data from {0} with {1} harmonic modes\n".format(args.input_filename, args.nmodes))
    asciitable.write(output,target, delimiter=args.delimiter)
    target.close()
    
