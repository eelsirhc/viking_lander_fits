#the original data containingl ls,vl1,vl2 data
baseline = $(wildcard data/planetWRF.run.data)
low = $(wildcard data/*low*.data)
high = $(wildcard data/*high*.data)

fit_data_low = $(addprefix fit/, $(notdir $(patsubst %.data,%.fit,$(low))))
fit_data_high = $(addprefix fit/, $(notdir $(patsubst %.data,%.fit,$(high))))
fit_baseline = $(addprefix fit/, $(notdir $(patsubst %.data,%.fit,$(baseline))))

viking_data = data/vl_p.dat
viking_fit = data/viking.fit

low_parameter_file = data/fit_low.parameters
high_parameter_file = data/fit_high.parameters

lander = vl1

viking_fit_low = fit/$(lander).low.fit
viking_fit_high = fit/$(lander).high.fit 

startrow=600
stoprow=-1
nmodes = 5

all: $(viking_fit_low) $(viking_fit_high)


$(viking_fit_low) :  $(fit_data_low) $(viking_fit) $(fit_baseline)
	@echo "Fitting parameters using low perturbation data"
	./fit_parameters.py --lander=$(lander) $(low_parameter_file) ${viking_fit} $@.parameter $@

$(viking_fit_high) : $(fit_data_high) $(viking_fit) $(fit_baseline)
	@echo "Fitting parameters using high perturbation data"
	./fit_parameters.py --lander=$(lander) $(high_parameter_file) ${viking_fit} $@.parameter $@

$(viking_fit) : $(viking_data)
	@echo "Fitting Viking data using $(nmodes) harmonic modes"
	@./fit_viking_data.py --vl1years=2,3 --vl2years=2 --nmodes=$(nmodes) $< $@

fit/%.fit : data/%.data
	@echo "Fitting $< using $(nmodes) harmonic modes" $< $@
	@./fit_to_ls_grid.py --startrow=$(startrow) --stoprow=$(stoprow) --nmodes=$(nmodes) $< $@

clean: 
	@rm -f  $(fit_data_low) \
		   	$(fit_data_high) \
			$(fit_data_baseline)\
			$(viking_fit)\
			$(viking_fit_low)\
			$(viking_fit_high)