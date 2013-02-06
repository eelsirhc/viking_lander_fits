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

Once all of the model simulations are complete and post-processed the data can be used in the fitting model to find the statistically optimum values of each parameter.

 [1]: https://github.com/ashima/planetWRF/commit/3017e40f894d23e3e4d432bd6b51ab0de98b0153