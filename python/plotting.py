import click
import core
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@click.group()
def plot():
    """plot command group

    Sets up the plot cli command.
    Args:
        N/A

    Returns:
        N/A

    Raises:
        N/A
    """
    pass


def csv(s):
    return [i for i in s.split(",")]

@plot.command()
@click.argument("filename",nargs=-1)
@click.option("--output", default="out.png")
@click.option("--lander", default="vl1")
@click.option("--base", default=None)
def data(filename, output="out.png", lander="vl1",base=None):
    data = dict()
    ls = np.arange(360)
    if base is not None:
        if base.endswith("fit"):
            base = core.fitter_load(base).interpolate(ls)
        else:
            base = core.read_file(base)
            ls = base["L_S"]
    else:
        base=0    
    
    cols = [c for c in base.columns if c!="L_S"]
    for f in filename:
        if f.endswith("fit"):
            data[f] = core.fitter_load(f).interpolate(ls)[cols].reset_index().rename(columns=dict(index="L_S"))
            data[f][cols] -= base[cols]
        else:
            data[f] = core.read_file(f)

    plt.figure(figsize=(8, 6))
    for f in filename:
        plt.plot(data[f]["L_S"], data[f][lander], label=f, marker="o", ls="")
    
    plt.xlabel("L_S")
    plt.ylabel("Pressure (Pa)")
    plt.legend()
    plt.savefig(output)

@plot.command()
@click.argument("filename")
@click.argument("fit_filename")
@click.argument("output_filename")
@click.option("--lander", default="vl1")
def data_fit(filename, fit_filename, output_filename, lander="vl1"):
    """plot something before I go crazy"""

    df = core.read_file(filename)
    fitter = core.fitter_load(fit_filename)

    y1 = fitter.interpolate(df["L_S"])   

    plt.figure(figsize=(8, 6))
    plt.plot(df["L_S"], df[lander], label="data", marker="o", ls="")
    plt.plot(df["L_S"], y1[lander], label="fit", marker="+", ls="")
#    plt.plot(df["L_S"], y2, label="fitconv", marker="+", ls="")
    plt.xlabel("L_S")
    plt.ylabel("Pressure (Pa)")
    plt.savefig(output_filename)


def register(com):
    com.add_command(plot)
