import click
import core
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@click.group()
def plot():
    pass


@plot.command()
@click.argument("filename")
@click.argument("output_filename")
@click.option("--lander", default="vl1")
def data(filename, output_filename, lander="vl1"):
    df = core.read_file(filename)
    plt.figure(figsize=(8, 6))
    plt.plot(df["L_S"], df[lander], label="data", marker="o", ls="")
    plt.savefig(output_filename)
    plt.xlabel("L_S")
    plt.ylabel("Pressure (Pa)")


@plot.command()
@click.argument("filename")
@click.argument("fit_filename")
@click.argument("output_filename")
@click.option("--lander", default="vl1")
def data_fit(filename, fit_filename, output_filename, lander="vl1"):
    df = core.read_file(filename)
    fit = core.read_file(fit_filename)
    conv = core.convert_AP_to_SC(fit).set_index("modes")[lander]
    y1 = core.fitfunc(df["L_S"].values, *fit[lander])
    print(conv)
    import pandas as pd

    y2 = core.calc_AP_SC(df["L_S"].values, pd.DataFrame(conv))

    plt.figure(figsize=(8, 6))
    plt.plot(df["L_S"], df[lander], label="data", marker="o", ls="")
    plt.plot(df["L_S"], y1, label="fit", marker="+", ls="")
    plt.plot(df["L_S"], y1, label="fitSC", marker="+", ls="")
    plt.savefig(output_filename)
    plt.xlabel("L_S")
    plt.ylabel("Pressure (Pa)")


def register(com):
    com.add_command(plot)
