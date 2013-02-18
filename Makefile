verbose=0
ifeq ($(verbose),1)
        quiet=
else
        quiet=@
endif
#directory names
fit = fit
data = data
#baseline is the base experiment about which the perturbations are taken
baseline = $(wildcard $(data)/base.data)
#low perturbations
low = $(wildcard $(data)/*low*.data)
#low perturbations
high = $(wildcard $(data)/*high*.data)

#viking lander data
viking_data = $(data)/vl_p.dat
viking_fit = $(fit)/viking.fit

#files describing the perturbations made in each experiment
low_parameter_file = $(data)/fit_low.parameters
high_parameter_file = $(data)/fit_high.parameters

#generated file names that convert data/filename.data -> fit/filename.fit
fit_data_low = $(addprefix $(fit)/, $(notdir $(patsubst %.data,%.fit,$(low))))
fit_data_high = $(addprefix $(fit)/, $(notdir $(patsubst %.data,%.fit,$(high))))
fit_baseline = $(addprefix $(fit)/, $(notdir $(patsubst %.data,%.fit,$(baseline))))

#default lander name
lander = vl1

#the final output filenam
viking_fit_low = $(fit)/$(lander).low.fitted
viking_fit_high = $(fit)/$(lander).high.fitted

#the start of the data to extract from the GCM output, used to skip initial relaxation
startrow=5200
stoprow=-10

#number of harmonic modes
#total number of modes generated is 1 + 2*nmodes
nmodes = 5
vmodes = 5

#default make rule
all: $(viking_fit_low) $(viking_fit_high)

#to make the low perturbation we require all of the polynomial fits 
#for the low perturbations and the baseline and viking fits
#Using these data we construct the Jacobian out of the perturbation experiments
#and the baseline, and use this to calculate the state vector that will
#minimize the least square error residual between the Viking lander pressure and
#the GCM pressure.
$(viking_fit_low) :  $(fit_data_low) $(viking_fit) $(fit_baseline)
	@echo "Fitting parameters using low perturbation data"
	$(quiet)./python/fit_parameters.py --lander=$(lander) $(low_parameter_file) ${viking_fit} $@.parameter $@

#as viking_fit_low, but for high perturbations
$(viking_fit_high) : $(fit_data_high) $(viking_fit) $(fit_baseline)
	@echo "Fitting parameters using high perturbation data"
	$(quiet)./python/fit_parameters.py --lander=$(lander) $(high_parameter_file) ${viking_fit} $@.parameter $@

#To construct the Viking fit data, we need to extract data from the PDS dataset stored in the data directory
#In this case we extract VL1 years 2 and 3, and VL2 year 2 only. This are fitted using the harmonic function.
$(viking_fit) : $(viking_data)
	@echo "Fitting Viking data using $(vmodes) harmonic modes"
	$(quiet)./python/fit_viking_data.py --vl1years=2,3 --vl2years=2 --nmodes=$(vmodes) $< $@

#Generic rule, for a required fit file, we need the equivalent data file
#we take the data file and fit the harmonic function of nmodes to it
fit/%.fit : data/%.data
	@echo "Fitting $< using $(nmodes) harmonic modes"
	$(quiet)./python/fit_to_ls_grid.py --startrow=$(startrow) --stoprow=$(stoprow) --nmodes=$(nmodes) $< $@

#remove fit files.
clean: 
	$(quiet)rm -f  $(fit_data_low) \
		   	$(fit_data_high) \
			$(fit_data_baseline)\
			$(viking_fit)\
			$(viking_fit_low)\
			$(viking_fit_high)\
			$(fit_baseline)\
			$(viking_fit_low).parameter\
			$(viking_fit_high).parameter
