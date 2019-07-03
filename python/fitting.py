import click
import numpy as np
import pandas as pd
import core
import os


@click.group(help="Fit command group")
def fit():
    """fit command group
    """
    pass


def fit_single_model(
    filename,
    output_filename,
    startrow=0,
    stoprow=None,
    delimiter=",",
    nmodes=5,
    fittype="AP",
):
    """Fit a single model

    Sets up the plot cli command.
    Args:
        filename : A file containing L_S and pressure data.
        output_filename : A file to store the resulting fitted parameters.
        startrow : The row of the input data to start from.
        stoprow : The row of the input data to end on.
        delimiter : The field delimiter in the input file
        nmodes : The number of fitted modes to use.
        fittype : The type of fit to use (AP = Amplitude&Phase, CS=Cosine&Sine, Window=fixed width window)
    Returns:
        N/A

    Raises:
        N/A
    """
    data = core.read_file(
        filename, startrow=startrow, stoprow=stoprow, delimiter=delimiter
    )
    fitter = core.fitter(fittype, nmodes=nmodes)
    _ = fitter.fit(data)
    fitter.write_file(output_filename)


@fit.command()
@click.argument("filename")
@click.option("--startrow", default=0, type=int,help="The row of the input data to start from.")
@click.option("--stoprow", default=None, type=int, help="The row of the input data to end on.")
@click.option("--delimiter", default=",", type=str, help="The field delimiter in the input file")
@click.option("--nmodes", default=5, type=int, help="The number of fitted modes to use.")
@click.option("--fittype", default="AP", type=str, help="""The type of fit to use (AP = Amplitude&Phase,
                                          CS=Cosine&Sine,
                                          Window=fixed width window)
    """)
def models(filename, startrow=0, stoprow=None, delimiter=None, nmodes=5, fittype="AP"):
    """Fit the models referred to in FILENAME in a column called 'name'.

    The FILENAME input file is typically the run_values file that contains the
    perturbed input values for each simulation of the form

    | name,massair_target,...
    | base,2.83e16,...
    | perturb,2.93e16,...

    The name column is used to read a data file with the same name in the data directory as data/name.data
    The output filename is automatically generated as fit/name.fit

    """
    runfile = pd.read_csv(filename, comment="#")
    for simulation in runfile.name:
        infilename = "data/{}.data".format(simulation)
        output_filename = "fit/{}.fit".format(simulation)
        fit_single_model(
            infilename,
            output_filename,
            startrow,
            stoprow,
            delimiter,
            nmodes,
            fittype=fittype,
        )


@fit.command()
@click.argument("filename")
@click.argument("output_filename")
@click.option("--startrow", default=0, type=int,help="The row of the input data to start from.")
@click.option("--stoprow", default=None, type=int, help="The row of the input data to end on.")
@click.option("--delimiter", default=",", type=str, help="The field delimiter in the input file")
@click.option("--nmodes", default=5, type=int, help="The number of fitted modes to use.")
@click.option("--fittype", default="AP", type=str, help="""The type of fit to use (AP = Amplitude&Phase,
                                          CS=Cosine&Sine,
                                          Window=fixed width window)""")
def model(
    filename,
    output_filename,
    startrow=0,
    stoprow=None,
    delimiter=None,
    nmodes=5,
    fittype="AP",
):
    """Fits a single model.

    Args:
        filename: A file containing model names corresponding to the directory
        output_filename: A file to store the resulting fitted parameters.
        startrow: The row of the input data to start from.
        stoprow: The row of the input data to end on.
        delimiter: The field delimiter in the input file
        nmodes: The number of fitted modes to use.
        fittype: The type of fit to use (AP = Amplitude&Phase,
                                          CS=Cosine&Sine,
                                          Window=fixed width window)
    Returns:
        N/A

    Raises:
        N/A
    """
    fit_single_model(
        filename, output_filename, startrow, stoprow, delimiter, nmodes, fittype
    )


def csvi(s):
    """Split a string into comma separated integer values

    Args:
        s : The string to split

    Returns:
        A list of integer values

    Raises:
        N/A
    """
    return [int(i) for i in s.split(",")]


@fit.command()
@click.argument("filename")
@click.argument("output_filename")
@click.option("--lander", default="VL1", type=str)
@click.option("--years", default=None, type=csvi)
@click.option("--nmodes", default=5, type=int)
@click.option("--fittype", default="AP", type=str)
def viking(filename, output_filename, lander, years, nmodes=5, fittype="AP"):

    if years is None:
        years = dict(VL1=[2, 3], VL2=[2])[lander]

    data = core.read_viking(filename, lander, years)

    fitter = core.fitter(fittype, nmodes=nmodes)
    _ = fitter.fit(data)
    fitter.fitdata = fitter.fitdata.rename(columns=dict(Pressure=lander.lower()))
    fitter.write_file(output_filename)


@fit.command()
@click.argument("parameter_filename", type=str)
@click.argument("viking_filename", type=str)
@click.option("--suffix", type=str, default="low")
def parameters(parameter_filename, viking_filename, suffix="low"):
    parameters = pd.read_csv(parameter_filename, comment="#", index_col=0)
    viking = core.fitter_load(viking_filename)

    data = []
    names = []
    delta_names = []
    lander = "vl1"
    ls = np.arange(0, 360, 1.0)
    target = viking.interpolate(ls)[lander]

    data.append(ls)
    names.append("L_S")
    for simulation in parameters.index:
        if simulation.endswith(suffix) or simulation == "base":
            print("loading {}".format(simulation))
            df = core.fitter_load("fit/{}.fit".format(simulation))
            names.append(simulation)
            delta_names.append(simulation)
            data.append(df.interpolate(ls)[lander].values)

    data = pd.DataFrame(np.array(data).T, columns=names)  # .set_index("L_S")

    dy = data.copy()
    for c in [x for x in delta_names if x != "base"]:
        dy[c] -= dy["base"]
    del dy["base"]

    dx = parameters.T.copy()
    for c in [x for x in delta_names if x != "base"]:
        dx[c] = dx[c] - dx["base"]
        if c == "massair":
            dx[c] /= dx["base"]
    del dx["base"]
    delta_names.remove("base")
    dx = dx[delta_names].T
    del dx["massair_target"]

    target = pd.DataFrame(target.values.T)  # , index=ls)
    target = target  # * data["base"].values.mean() / target.values.mean()
    delta = target.values.squeeze() - data["base"].values
    popt, pstd, dydx = core.least_square_inversion(
        dy[delta_names].values, dx.values, delta
    )
    print(popt)

    # output the parameters

    for i, col in enumerate(dx.columns):
        arr = dx[col]
        row = arr[arr != 0].index.values[0]
        print(col, row)

        line = [
            col,
            parameters.loc[row, col],
            popt[i],
            parameters.loc[row, col] + popt[i],
        ]
        # resx = x[c]
        # line = [c, res[res != 0].values[0], resx[resx!=0].values[0]]

        print(line)


def register(com):
    com.add_command(fit)
