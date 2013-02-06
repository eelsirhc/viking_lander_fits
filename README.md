To calibrate the ice model and base atmospheric mass in WRF we fit the ice parameters so that the pressure cycle matches the Viking Lander pressure curves. In the current model the parameters are the *total atmospheric mass*, and the co2 ice *albedo* and *emissivity* in both hemispheres (4 parameters)

### Typical Values

From prior experiments (Guo et al., 2009), the best fit values for WRF 3.0.1.2 were

    co2mass = 2.8505e16
    sh_albedo = 0.461
    sh_emiss = 0.785
    nh_albedo = 0.795
    nh_emiss = 0.485
    

### Fitting procedure

To generate these fits, a linear(izable) model is assumed to exist for the relationship between the pressure value measured at the Viking Lander sites (notably, **VL1**) and the 5 parameter values defined above. The model is then generated as

    P(x) = P(x0) + dP/dx . dx 

where x0 represents the state vector of the five parameters above, P(x) is the surface pressure, and P(x0) a reference base state about which we linearize. If our target P(x) is the observed Viking lander pressure, then

    P(x) - P(x0) = dP/dx . dx
    deltaP = Jp . dx

and deltaP is the difference between our base state and our target state, J_p is the sensitivity matrix (the *Jacobian*) relating changes in the state vector to changes in the pressure, and dx is the change in the state vector from the reference state required to produce the deltaP.

We can find Jp by running 6 simulations, including 1 baseline experiment and 1 perturbed experiment for each state parameter. The baseline experiment provides dual service as P0 and in Jp as

    Jp = (P' - P0)/(x' - x0)


### Setup

Using the current version of WRF 331 (at least changeset [@3017e40][1]) the namelist allows you to specify each of the five parameters above (saving an annoying recompilation for every change). To automate the experiment a bit I create a template directory containing

    Data
    ideal.single
    namelist.input
    runscript.wrf
    transfer.log
    wrf.exe
 
where *ideal.single* is a serial version of ideal.exe (and is less annoying to run on the NASA cluster).

A *run_values* file is created to define the experiment names and parameter values

    name,massair_target,sh_co2ice_albedo,sh_co2ice_emiss,nh_co2ice_albedo,nh_co2ice_emiss
    base,          2.939e16, 0.461, 0.785, 0.795, 0.485
    co2_low,       2.79e16,  0.461, 0.785, 0.795, 0.485
    sh_albedo_low, 2.939e16, 0.431, 0.785, 0.795, 0.485
    sh_emiss_low,  2.939e16, 0.461, 0.755, 0.795, 0.485
    nh_albedo_low, 2.939e16, 0.461, 0.785, 0.765, 0.485
    nh_emiss_low,  2.939e16, 0.461, 0.785, 0.795, 0.455
    co2_high,      3.08e16,  0.461, 0.785, 0.795, 0.485
    sh_albedo_high,2.939e16, 0.491, 0.785, 0.795, 0.485
    sh_emiss_high, 2.939e16, 0.461, 0.815, 0.795, 0.485
    nh_albedo_high,2.939e16, 0.461, 0.785, 0.825, 0.485
    nh_emiss_high, 2.939e16, 0.461, 0.785, 0.795, 0.515


Note that this setup runs a *positive perturbation* experiment (*high* values) and a *negative perturbation* experiment. If the above model is linear (or very nearly linear), these experiments will produce the same answer, if there are non-linearities then we might expect different feedback response for a positive and negative perturbation (e.g. reducing total mass might push the atmosphere below a threshold for seasonal collapse that increasing the mass would never reach).

This experiment list is used to generate 11 directories and run the experiments with the python script called *runscript.py* in the *simulations* directory. The result of this code will be to create all 11 directories, and print to **STDOUT** a startup runscript to start each simulation (on Pleiades). Either pipe this script to a file and execute or copy the screen output and paste into the terminal.

The key to perturbing each experiment is the format of the *namelist.input* file. An extract of which looks like

    sh_co2ice_albedo                    = {sh_co2ice_albedo}
    sh_co2ice_emiss                     = {sh_co2ice_emiss}
    nh_co2ice_albedo                    = {nh_co2ice_albedo}
    nh_co2ice_emiss                     = {nh_co2ice_emiss}
    /

The braced words are used by python and replaced with values. Each *row* of the asciitable *data* is converted into a dictionary with keys matching the table header, and values from each data row. In the script the namelist is first read into a string, then the string is *format*ted by being given this dictionary as keyword arguments (the .format(**d_row) incantation). For each braced string in the namelist file, an equivalently named key is found and its value inserted in the namelist string, which is then written back to the file.

then you wait...

### Post-processing

After the simulations are complete, we need to extract the surface pressure at the Viking Lander locations, and the *aerocentric longitude* L_s. The Python script viking_lander.py can be used to perform this calculation on multiple files if called as

    python viking_lander.py list_of_input_files --output output_file
    (e.g.) 
    python viking_lander.py wrfout* --output vl.data

This will produce an ascii data file containing three comma separated columns of L_s, adjusted surface pressure at VL1, and adjusted surface pressure at VL2. For each time sample in the file, the code reads in the surface pressure nearest to the lander sites, bilinearly interpolates to the landing site location, and adjusts the surface pressure from the model surface altitude to the 'known' lander altitude to account for the relatively low model resolution.

### Fitting

Once all of the model simulations are complete and post-processed the data can be used in the fitting model to find the statistically optimum values of each parameter. To fit the model parameters we take five perturbation simulations, the reference baseline, and the target pressure cycle and calculate the Jacobian (J) above using the 5 perturbation simulations, the target deltaP from the target pressure and reference pressure. Then we use a Python subroutine to calculate the parameters that minimize the sum of the squared errors between between the target deltaP and the J.dx. 

The Python code that performs this minimization is separated into three programs. Two to find the harmonic fits to the simulated *or* observed pressure cycle ( `fit_to_ls_grid.py` and `fit_viking_data.py` ), and one to calculate the state vector that gives the LMSE residual ( `fit_parameters.py` ) given these harmonic fits. A `Makefile` is used to control the fitting process to reduce the duplication of the calling the fitting routines. Calling `make` alone should produce a fit to the Viking Lander 1 year 2&3 data if the source data is available.

The harmonic fitting functions are essentially the same, but deal with two formats. The format output by the simulation post-processing code ( `viking_lander.py` ) is processed by the `fit_to_ls_grid.py` code, which reads in the entire file, optionally using only a segment of the code (given by `--startrow` and `--stoprow ) and fits a function composed of a mean value and `--nmodes` harmonic modes. The result of calling this function will be a new **fit** file with three columns, the mode ordinal number (from 0 to 2\*nmodes ) where n=1,3,5,7,9 are the amplitudes of the sin((n+1)/2) mode, and n=2,4,6,8,10 are the phases in degrees of the n/2 mode. The second and third column contain the fitted parameter to the VL1 and VL2 data from the input file.

    e.g. 
	./python/fit_to_ls_grid.py --startrow=600 --stoprow=1200 --nmodes=5 input.data output.fit

will read the data from *input.data* extract lines 600:1200, fit a 5 harmonic function and output the results in output.fit

The Viking lander code in `fit_viking_data.py` performs the same function, but reads the particular form of the vl\_p.dat file from [PDS][2]. In this format, the VL1 and VL2 datasets are in the same format, with the lander name as one of the columns, and pressure is given in mbars (or equivalently hPa), instead of Pa as in the GCM. The `fit_viking_data.py` optionally takes a comma separated list of years in `--vl1years` and `--vl2years` to select different years to fit. The default of `--vl1years=2,3 --vl2years=2` is chosen to use three relatively dust-storm free data.
 
    e.g.
	./python/fit_viking_data.py --vl1years=2,3 --vl2years=2 --nmodes=5 input.pds output.fit

extracts the appropriate years of data, performs the fit with 5 harmonic modes and save the result in output.fit in the same format as the fitted simulation data.

To call the parameter fitting code `fit_parameters.py` a file containng the experimental setup is needed that gives the location of the fitted data, and the parameters used in the simulation in order to calculate the Jacobian denominator. An example of this file for the low perturbation runs is

    filename, co2, sh_albedo, sh_emiss, nh_albedo, nh_emiss
    fit/planetWRF.run.fit,                1.0, 0.461, 0.785, 0.795, 0.485
    fit/planetWRF.co2_low.run.fit,        0.965, 0.461, 0.785, 0.795, 0.485
    fit/planetWRF.sh_albedo_low.run.fit,  1.0, 0.431, 0.785, 0.795, 0.485
    fit/planetWRF.sh_emiss_low.run.fit,   1.0, 0.461, 0.755, 0.795, 0.485
    fit/planetWRF.nh_albedo_low.run.fit,  1.0, 0.461, 0.785, 0.765, 0.485
    fit/planetWRF.nh_emiss_low.run.fit,   1.0, 0.461, 0.785, 0.795, 0.435

where the first row gives the column names that correspond to the parameters being perturbed, and the body of the data contains the actual values of parameters used (not their perturbations). This file is used to read in each *.fit* file, construct a deltaP and deltaX from the base state (the first row), and J=deltaP/deltaX . The Python least square fitting algorithm **scipy.linalg.lstsq** is then used to find the LMSE optimum state vector perturbation dx . This state vector perturbation is used to construct a best fit pressure cycle (and residuals) for both VL1 and VL2, which are output in a final *.fit* file with the new parameter values contained in the header, and the fitted pressure cycle in the body of the table, e.g. called as

    ./python/fit_parameters.py --lander=vl1 data/fit_low.parameters fit/viking.fit output.parameter output

produces a files similar to this

    #Best fit pressure curve to vl1 data
    #Parameter fit to best reproduce vl1 data
    #names,basevalue,perturbation,newvalue
    #co2,1.0,0.0148,1.0148
    #sh_albedo,0.461,-0.0347,0.4263
    #sh_emiss,0.785,0.1456,0.9306
    #nh_albedo,0.795,-0.0354,0.7596
    #nh_emiss,0.485,-0.3009,0.1841
	#NOTE THESE EXAMPLE VALUES ARE NOT USEFUL
    L_S,vl1,vl2,res_vl1,res_vl2
    0,812.505565331,901.793299665,-8.28206091811,-19.2443976334
    1,812.742683939,902.915796375,-8.33971138839,-20.1599371335
    ....

The output.parameter in the above example contains just the parameter table (that appears in the header above) as a readable table without the fitted pressure data.

### Makefile
To simplify the post-processing, the makefile in the main directory can be used to automatically run this code (if the input filenames are correct). Called `make clean` will remove all generated files, `make all` will perform a fit of VL1 data using the low and high perturbations, `make -j 8 all -e lander=vl2` will do the same for VL2 data using 8 processes. Calling `make -B fit/planetWRF.run.fit` will unconditionally remake the harmonic fit of the base run. The simplest use of `make` will run the default rule (equivalent to `make all`) that will fit the VL1 data using low and high perturbation experiments.

 [1]: https://github.com/ashima/planetWRF/commit/3017e40f894d23e3e4d432bd6b51ab0de98b0153
 [2]: http://starbase.jpl.nasa.gov/vl1-m-met-4-binned-p-t-v-corr-v1.0/vl_1001/data/