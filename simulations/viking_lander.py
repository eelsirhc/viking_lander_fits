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
    b = p[:,1,0]*(1-fx) + p[:,1,1]*fx
    c = a*(1-fy) + b*fy

    return c
    
def func_pressure_curve(nc, index, loc, alldata=False):
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
    if alldata:
      return dict(original_psfc=psfc_loc, 
                  corrected_psfc=corrected_psfc,
                  hgt=hgt_loc,
                  tsfc=tsfc_loc,
                  dp=dp,
                  rho=rho)  
    return corrected_psfc

def func_vl1_pressure_curve(nc, index):
    """Calculates the surface pressure and Viking Lander 1"""
    loc  = {"lat":22.2692, "lon":-48.1887, "height":-3627.}
    return func_pressure_curve(nc, index, loc)
    
def func_vl2_pressure_curve(nc, index):
    """Calculates the surface pressure and Viking Lander 2"""
    loc = {"lat": 47.6680, "lon": 134.0430, "height": -4505.0}
    return func_pressure_curve(nc, index, loc)

def func_mpf_pressure_curve(nc, index):
    """Calculates the surface pressure and MPF"""
    loc = {"lat": 19.0949, "lon": -33.4908, "height": -3682.0}
    return func_pressure_curve(nc, index, loc)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="viking_lander_pressure.data")
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()

    data = []
    index=slice(None,None,None) ##all data in file
    
    output=open(args.output,'w')
    output.write("L_S,vl1,vl2,mpf\n") #header
    for filename in args.filenames:
        print (filename)
        nc = netCDF4.Dataset(filename)
        ls = nc.variables["L_S"][:]
        vl1 = func_vl1_pressure_curve(nc, index)
        vl2 = func_vl2_pressure_curve(nc, index)
        mpf = func_mpf_pressure_curve(nc, index)

        data = dict(L_S=ls, vl1=vl1, vl2=vl2, mpf=mpf)
        for i_ls, i_vl1, i_vl2, i_mpf in zip(ls, vl1, vl2, mpf):
            output.write("{0},{1},{2},{3}\n".format(i_ls, i_vl1, i_vl2, i_mpf))
        del nc #closes
    output.close()
    
