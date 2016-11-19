#!/usr/bin/env python
import netCDF4
import sys
import numpy
import argparse

# stuff for getting L_S from Times
zero_date=488.7045
planet_year=669
obliquity=25.19
eccentricity=0.09341233
small_value = 1e-6

equinox_fraction=None
if equinox_fraction is None:
    equinox_fraction = (planet_year - zero_date) / planet_year

deleqn = equinox_fraction * planet_year
obecl = numpy.deg2rad(obliquity)
sinob = numpy.sin(obecl)

er = numpy.sqrt((1.0 + eccentricity)/(1.0 - eccentricity))

def get_eq():
    """Gets anomaly at equinox as part of getting L_S as function of Times"""
#mean anomaly at equinox
    qq = 2.0 * numpy.pi * deleqn / planet_year

#anomaly at equinox eq
    e = 1.0
    cd0 = 1.
    while (cd0 > small_value):
        ep = e - (e - eccentricity*numpy.sin(e) - qq)/(1.-eccentricity*numpy.cos(e))
        cd0 = abs(e-ep)
        e=ep
    eq = 2. * numpy.arctan(er * numpy.tan(0.5*e))
 
    return eq

def marswrf_ls(julian):
    """Returns L_S given a day-of-year"""
#calculate longitude of sun
    dp_date = julian - zero_date
    while dp_date < 0:
        dp_date = dp_date + planet_year
    while dp_date > planet_year:
        dp_date = dp_date - planet_year

    eq = get_eq()

#true anomaly at current date
# em is the mean anomaly
# ep is the eccentric anomaly
    em = 2.0 * numpy.pi * dp_date / planet_year
    e=1.0
    cd0 =1.0

    while (cd0 > small_value):
        ep = e - (e - eccentricity*numpy.sin(e) - em)/(1.-eccentricity*numpy.cos(e))
        cd0 = abs(e-ep)
        e=ep
    w = 2.0 * numpy.arctan(er * numpy.tan(0.5*e))

    als = numpy.rad2deg(w-eq) #aerocentric longitude
    while(als < 0): 
        als+=360
    while(als) > 360:
        als-=360

    return als

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
    
def func_pressure_curve(nc, nc_i, index, loc, alldata=False):
    """Given the location dictionary (lon,lat, height), calculates
    the surface pressure at the location and adjust the value from the
    model surface height to the 'true' provided height"""
    
    lon = numpy.squeeze(nc_i.variables["XLONG"][0,0,:])
    lat = numpy.squeeze(nc_i.variables["XLAT"][0,:,0])
    psfc = nc.variables["PSFC"][index]
    psfc_loc = interp_to_site(lon, lat, psfc, loc["lat"],loc["lon"])
    
    if "height" not in loc:
        return psfc_loc
    
    rd    = nc.getncattr("R_D")
    cp    = nc.getncattr("CP")
    grav  = nc.getncattr("G")
    tsfc = nc.variables["TSK"][index]
    hgt = nc_i.variables["HGT"][index]
    
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

def func_vl1_pressure_curve(nc, nc_i, index):
    """Calculates the surface pressure at Viking Lander 1"""
    loc  = {"lat":22.2692, "lon":-48.1887, "height":-3627.}
    return func_pressure_curve(nc, nc_i, index, loc)
    
def func_vl2_pressure_curve(nc, nc_i, index):
    """Calculates the surface pressure at Viking Lander 2"""
    loc = {"lat": 47.6680, "lon": 134.0430, "height": -4505.0}
    return func_pressure_curve(nc, nc_i, index, loc)

def func_mpf_pressure_curve(nc, nc_i, index):
    """Calculates the surface pressure at MPF"""
    loc = {"lat": 19.0949, "lon": -33.4908, "height": -3682.0}
    return func_pressure_curve(nc, nc_i, index, loc)

def func_pho_pressure_curve(nc, nc_i, index):
    """Calculates the surface pressure at PHX"""
    loc = {"lat": 68.22, "lon": -125.75, "height": -4130.}
    return func_pressure_curve(nc, nc_i, index, loc)
    
def func_msl_pressure_curve(nc, nc_i, index):
    """Calculates the surface pressure at MSL"""
    loc = {"lat": -4.50846, "lon": 137.4416, "height": -4500.97}
    return func_pressure_curve(nc, nc_i, index, loc)
    
def func_get_ls_from_times(nc):
    """Return Ls given the netcdf file handle that contains the Times variable"""
    Times = nc.variables["Times"][:]
# assuming form: ['0' '0' '0' '3' '-' '0' '0' '0' '0' '2' '_' '0' '0' ':' '0' '0' ':' '0' '0']
    whole_day = ( (Times[:,5].astype(numpy.float))*10000. + (Times[:,6].astype(numpy.float))*1000. +
                  (Times[:,7].astype(numpy.float))*100.   + (Times[:,8].astype(numpy.float))*10. +
                  (Times[:,9].astype(numpy.float)) )
    part_day = ( (Times[:,11].astype(numpy.float)*10. + Times[:,12].astype(numpy.float))/24. + 
                 (Times[:,14].astype(numpy.float)*10. + Times[:,15].astype(numpy.float))/1440. +
                 (Times[:,17].astype(numpy.float)*10. + Times[:,18].astype(numpy.float))/86400. )
    day = whole_day + part_day
    times_in_file = numpy.size(day)
    ls = numpy.zeros_like(day)
    for i in range(0,times_in_file):
      ls[i] = marswrf_ls(day[i])
    return ls

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mars_lander_pressure.data")
    parser.add_argument("--use_wrfinput", action="store_true")
    parser.add_argument("--get_ls_from_Times", action="store_true")
    parser.add_argument("--average_over")
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()

    if args.use_wrfinput:
           print "... using wrfinput to get XLAT,XLONG,HGT"
    if args.get_ls_from_Times:
           print "... getting L_S from Times variable"
    if args.average_over is not None:
           print "... averging each output over ",int(args.average_over)," input entries"
    data = []
    index=slice(None,None,None) ##all data in file
    
    output=open(args.output,'w')
    output.write("L_S,vl1,vl2,mpf,phx,msl\n") #header
    for filename in args.filenames:
        print (filename)
        nc = netCDF4.Dataset(filename)
        if args.use_wrfinput:
           nc_i = netCDF4.Dataset('wrfinput_d01')
        else:
           nc_i = nc
        if args.get_ls_from_Times:
           ls = func_get_ls_from_times(nc)
#           print "[DEBUG] first ls in file: ",ls[0]
        else:
           ls = nc.variables["L_S"][:]
        vl1 = func_vl1_pressure_curve(nc, nc_i, index)
        vl2 = func_vl2_pressure_curve(nc, nc_i, index)
        mpf = func_mpf_pressure_curve(nc, nc_i, index)
        pho = func_pho_pressure_curve(nc, nc_i, index)
        msl = func_msl_pressure_curve(nc, nc_i, index)

        if args.average_over is not None:
           new_ls  = numpy.mean(ls.reshape(-1,int(args.average_over)), axis=1)
           new_vl1 = numpy.mean(vl1.reshape(-1,int(args.average_over)), axis=1)
           new_vl2 = numpy.mean(vl2.reshape(-1,int(args.average_over)), axis=1)
           new_mpf = numpy.mean(mpf.reshape(-1,int(args.average_over)), axis=1)
           new_pho = numpy.mean(pho.reshape(-1,int(args.average_over)), axis=1)
           new_msl = numpy.mean(msl.reshape(-1,int(args.average_over)), axis=1)

           data = dict(L_S=new_ls, vl1=new_vl1, vl2=new_vl2, mpf=new_mpf, pho=new_pho, msl=new_msl)
           for i_ls, i_vl1, i_vl2, i_mpf, i_pho, i_msl in zip(new_ls, new_vl1, new_vl2, new_mpf, new_pho, new_msl):
               output.write("{0},{1},{2},{3},{4},{5}\n".format(i_ls, i_vl1, i_vl2, i_mpf, i_pho, i_msl))

        else:
           data = dict(L_S=ls, vl1=vl1, vl2=vl2, mpf=mpf, pho=pho, msl=msl)
           for i_ls, i_vl1, i_vl2, i_mpf, i_pho, i_msl in zip(ls, vl1, vl2, mpf, pho, msl):
               output.write("{0},{1},{2},{3},{4},{5}\n".format(i_ls, i_vl1, i_vl2, i_mpf, i_pho, i_msl))

        del nc #closes
        del nc_i #closes
    output.close()
    
