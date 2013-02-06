#!/usr/bin/env python
import netCDF4
import sys
import numpy
import argparse

def interp_to_site(lon, lat, data, tolat, tolon):
    """Interpolates a dataset to a single longitude and latitude point"""
    wlon = numpy.where((lon[:-1] <= tolon)&(lon[1:]>tolon))
    wlon = wlon[0][0]
    wlat = numpy.where((lat[:-1] <= tolat)&(lat[1:]>tolat))
    wlat = wlat[0][0]
    
    p = data[:,wlat:wlat+2,wlon:wlon+2]
    fx = (tolon-lon[wlon])/(lon[wlon+1]-lon[wlon])
    fy = (tolat-lat[wlat])/(lat[wlat+1]-lat[wlat])
    a = p[:,0,0]*(1-fx) + p[:,0,1]*fx
    b = p[:,1,1]*(1-fx) + p[:,1,1]*fx
    c = a*(1-fy) + b*fy

    return c
    
def func_pressure_curve(nc, index, loc):
    """Given the location dictionary (lon,lat, height), calculates
    the surface pressure at the location and adjust the value from the
    model surface height to the 'true' provided height"""
    
    lon = numpy.squeeze(nc.variables["XLONG"][0,0,:])
    lat = numpy.squeeze(nc.variables["XLAT"][0,:,0])
    psfc = nc.variables["PSFC"][index]
    psfc_loc = interp_to_site(lon, lat, psfc, loc["lat"],loc["lon"])
    
    if "height" not in loc:
        return psfc_loc
    
    rd    = nc.getncattr("R_D")
    cp    = nc.getncattr("CP")
    grav  = nc.getncattr("G")
    tsfc = nc.variables["TSK"][index]
    hgt = nc.variables["HGT"][index]
    
    tsfc_loc = interp_to_site(lon, lat, tsfc, loc["lat"],loc["lon"])
    hgt_loc  = interp_to_site(lon, lat, hgt,  loc["lat"],loc["lon"])

    rho = psfc_loc/(rd*tsfc_loc)
    dp  = -rho*grav*(loc["height"]-hgt_loc)
    
    corrected_psfc = psfc_loc+dp
    return corrected_psfc

def func_vl1_pressure_curve(nc, index):
    """Calculates the surface pressure and Viking Lander 1"""
    loc  = {"lat":22.5, "lon":-50, "height":-3627.0}
    return func_pressure_curve(nc, index, loc)
    
def func_vl2_pressure_curve(nc, index):
    """Calculates the surface pressure and Viking Lander 2"""
    loc = {"lat": 48.3, "lon": 134.1, "height": -4505.0}
    return func_pressure_curve(nc, index, loc)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="viking_lander_pressure.data")
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()

    data = []
    index=slice(None,None,None) ##all data in file
    
    output=open(args.output,'w')
    output.write("Ls,vl1,vl2\n") #header
    for filename in args.filenames:
        print (filename)
        nc = netCDF4.Dataset(filename)
        ls = nc.variables["L_S"]
        vl1 = func_vl1_pressure_curve(nc, index)
        vl2 = func_vl2_pressure_curve(nc, index)
        for i_ls, i_vl1, i_vl2 in zip(ls, vl1, vl2):
            output.write("{0},{1},{2}\n".format(i_ls, i_vl1, i_vl2))
        del nc #closes
    output.close()
    
