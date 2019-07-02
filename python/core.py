# core
import pandas as pd
from scipy.optimize import curve_fit, leastsq
import numpy as np


def fitfunc(x, *p):
    """Model function, defined as y = mean + sum( amp[i]*cos( 2*pi*mode*(t-phase[i])/360. ) )"""
    val = p[0]
    for n in range(0, len(p) - 1, 2):
        ind = n + 1
        mode = (n / 2) + 1
        val = val + p[ind] * np.cos(2 * np.pi * mode * (x - p[ind + 1]) / 360.0)
    return val


def convert_AP_to_SC(data):
    """Convert amplitude phase cosine waves to cosine, sine."""
    # find the number of nodes
    modes = [int(d[5:]) for d in data.modes if d.startswith("Amode")]
    df = data.set_index("modes").T
    for mode in [d for d in data.modes if d.startswith("Amode")]:
        m = int(mode[5:])
        df["Smode{:02d}".format(m)] = df["Amode{:02d}".format(m)] * np.sin(
            m * np.deg2rad(df["Pmode{:02d}".format(m)])
        )
        df["Cmode{:02d}".format(m)] = df["Amode{:02d}".format(m)] * np.cos(
            m * np.deg2rad(df["Pmode{:02d}".format(m)])
        )
        del df["Amode{:02d}".format(m)]
        del df["Pmode{:02d}".format(m)]
    df = df.T.reset_index()
    return df


def calc_AP_SC(ls, data):
    df = data.T
    y = np.zeros_like(ls) + df["Mean"].values

    for mode in [d for d in data.index if d.startswith("Smode")]:
        m = int(mode[5:])
        y = y + (
            df["Smode{:02d}".format(m)].values * np.sin(m * np.deg2rad(ls))
            + df["Cmode{:02d}".format(m)].values * np.sin(m * np.deg2rad(ls))
        )
    return y


def read_viking(filename, lander, years):
    df = pd.read_csv(
        filename,
        delimiter="\s+",
        engine="python",
        names=["lander", "year", "L_S", "sol", "hour", "minute", "second", "Pressure"],
    )
    df = df[df["lander"] == lander]
    sel = np.zeros(len(df), dtype=bool)
    sel[:] = False
    for y in years:
        sel = sel | (df["year"] == y)
    df = df[sel][["L_S", "Pressure"]]
    df["Pressure"] *= 100
    return df


def read_file(filename, startrow=0, stoprow=None, delimiter=None):
    """Reads the data file from the filename, optionally skipping number of rows,
    using an optional delimiter """

    data = pd.read_csv(filename, delimiter=delimiter)
    sl = slice(startrow, stoprow, None)
    data = data.iloc[sl].copy()
    return data


def fit_data(data, nmodes):
    npars = 1 + 2 * nmodes
    row_names = ["Mean"]
    for i in range(1, nmodes + 1):
        row_names.extend(["Amode{:02d}".format(i), "Pmode{:02d}".format(i)])

    p0 = np.zeros(npars, dtype=np.float64)
    p1 = None
    vl1 = None
    p2 = None
    vl2 = None
    L_S = np.arange(360)
    fit = []
    use_columns = [x for x in data.columns if x != "L_S"]
    for col in use_columns:
        popt, pcov = curve_fit(
            fitfunc,
            data["L_S"].values,
            data[col].values,
            p0=p0,
            absolute_sigma=True,
            sigma=np.ones_like(data["L_S"].values) + 5,
        )
        pstd = np.sqrt(np.diag(pcov))
        fit.append(popt)
    return pd.DataFrame(np.array(fit).T, columns=use_columns, index=row_names)
    # return fit


def least_square_inversion(dy, dx, B):
    """Invert Ax=B where A=dy/dx, B=B."""
    if any((dx > 0).sum(axis=1).values > 1):
        raise ValueError("Unclear perturbations.")

    dx_column = dx.max(axis=1)
    dydx = dy.copy()
    for c in dydx.columns:
        dydx[c] = dy[c] / dx_column[c]

    def fitx(x, *p):
        val = 0.0
        for n, m in zip(p, x):
            val = val + n * m
        return val

    p0 = np.zeros(dx_column.size) + 0.1
    popt, pcov = curve_fit(fitx, dydx.values.T, B.values, p0=p0)

    pstd = np.sqrt(np.diag(pcov))

    return popt, pstd, dydx
