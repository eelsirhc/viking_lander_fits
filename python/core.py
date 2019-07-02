# core
import pandas as pd
from scipy.optimize import curve_fit, leastsq
import numpy as np
import os


class FitObject(object):
    def __init__(self, type=None):
        self.type = type
        self.fitdata = None
        self.index_label = ""

    def _load(self, filename):
        self.fitdata = read_file(filename)

    def fit(self, data):
        p0 = np.zeros(self.npars, dtype=np.float64)
        p1 = None
        fitdata = []
        fitstd = []
        use_columns = [x for x in data.columns if x != "L_S"]
        for col in use_columns:
            popt, pcov = curve_fit(
                self.model,
                data["L_S"].values,
                data[col].values,
                p0=p0,
                absolute_sigma=True,
                sigma=np.ones_like(data["L_S"].values) + 5,
            )
            pstd = np.sqrt(np.diag(pcov))
            fitdata.append(popt)
            fitstd.append(pstd)
        self.fitdata = pd.DataFrame(
            np.array(fitdata).T, columns=use_columns, index=self.row_names
        )
        self.fitstd = pd.DataFrame(
            np.array(fitstd).T, columns=use_columns, index=self.row_names
        )

        return self.fit

    def interpolate(self, ls):
        use_columns = [x for x in self.fitdata.columns if x != "L_S" and x != "modes"]
        intdata = []

        for col in use_columns:
            d = self.model(ls, *self.fitdata[col].values)
            intdata.append(d)
        result = pd.DataFrame(np.array(intdata).T, columns=use_columns)
        result["L_S"] = ls
        return result

    def write_file(self, output_filename):
        if self.fit is not None:
            # output the file
            if not os.path.exists(os.path.dirname(output_filename)):
                os.makedirs(os.path.dirname(output_filename))
            self.fitdata.to_csv(output_filename, index_label=self.index_label)
        else:
            raise ValueError("Run Fit first")

    def convert(self, other):
        """convert to another type"""
        pass

    def __getitem__(self, selector):
        return self.fitdata[selector]


def fitfunc_SC(x, *p):
    """Model function, defined as y = mean + sum( amp[i]*cos( 2*pi*mode*(t-phase[i])/360. ) )"""
    val = p[0]
    for n in range(0, len(p) - 1, 2):
        ind = n + 1
        mode = (n / 2) + 1
        val = (
            val
            + p[ind] * np.sin(2 * np.pi * mode * (x) / 360.0)
            + p[ind + 1] * np.cos(2 * np.pi * mode * (x) / 360.0)
        )
    return val


class FitCS(FitObject):
    """Fits to Sines and Cosines."""

    def __init__(self, nmodes=5):
        super(FitObject, self).__init__()
        self.nmodes = nmodes
        self.npars = 1 + 2 * nmodes
        self.row_names = ["Mean"]
        for i in range(1, nmodes + 1):
            self.row_names.extend(["Cmode{:02d}".format(i), "Smode{:02d}".format(i)])
        self.model = fitfunc_SC
        self.index_label = "modes"

    def load(self, filename):
        self._load(filename)
        self.nmodes = len([d for d in self.fitdata["modes"] if d.startswith("Smode")])

    def convert(self, other):
        pass


def fitfunc_AP(x, *p):
    """Model function, defined as y = mean + sum( amp[i]*cos( 2*pi*mode*(t-phase[i])/360. ) )"""
    val = p[0]
    for n in range(0, len(p) - 1, 2):
        ind = n + 1
        mode = (n / 2) + 1
        val = val + p[ind] * np.cos(2 * np.pi * mode * (x - p[ind + 1]) / 360.0)
    return val


class FitAP(FitCS):
    """Fits Amplitude & Phase."""

    def __init__(self, nmodes=5):
        super(FitObject, self).__init__()
        self.nmodes = nmodes
        self.npars = 1 + 2 * nmodes
        self.row_names = ["Mean"]
        for i in range(1, nmodes + 1):
            self.row_names.extend(["Amode{:02d}".format(i), "Pmode{:02d}".format(i)])
        self.model = fitfunc_AP
        self.index_label = "modes"


class FitWindow(FitObject):
    def __init__(self, nmodes=5):
        super(FitObject, self).__init__()
        self.nmodes = nmodes
        self.npars = nmodes
        self.index_label = "windows"
        self.nmodes = nmodes

    def fit(self, data):
        df = data.copy()
        df["windows"] = pd.cut(
            data["L_S"], np.linspace(0, 360, self.nmodes), labels=False
        )
        self.fitdata = df.groupby("windows").mean()

        return self.fitdata

    def load(self, filename):
        self._load(filename)
        self.nmodes = len([d for d in self.fitdata["windows"]]) + 1

    def interpolate(self, ls):
        use_columns = [x for x in self.fitdata.columns if x != "L_S" and x != "modes"]
        intdata = []

        for col in use_columns:
            d = np.interp(ls, self.fitdata["L_S"], self.fitdata[col])
            intdata.append(d)
        result = pd.DataFrame(np.array(intdata).T, columns=use_columns)
        result["L_S"] = ls
        return result


def fitter(fittype, *args, **kwargs):
    f = dict(CS=FitCS, AP=FitAP, Window=FitWindow)
    return f[fittype](*args, **kwargs)


def fitter_load(filename, *args, **kwargs):
    fitdata = read_file(filename)
    ap = 0
    cs = 0

    if "modes" in fitdata.columns:
        cs = len([d for d in fitdata["modes"] if d.startswith("Smode")])
        ap = len([d for d in fitdata["modes"] if d.startswith("Amode")])
    if (ap > 0) and (cs > 0):
        raise ValueError("Conflicting data in fit file {}".format(filename))
    elif ap > 0:
        fittype = "AP"
    elif cs > 0:
        fittype = "CS"
    else:
        fittype = "Window"

    f = fitter(fittype, *args, **kwargs)
    f.load(filename)
    return f


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


def least_square_inversion(dy, dx, B):
    """Invert Ax=B where A=dy/dx, B=B."""
    if any((dx != 0).sum(axis=1) > 1):
        raise ValueError("Unclear perturbations.")

    dx_column = np.array(
        [col[c] for col in dx.T for c in range(col.shape[0]) if col[c] != 0]
    )
    dydx = dy.copy()
    dydx /= dx_column[None, :]
    #print("DYDX=",dydx)
    # for c in dydx:
    #    dydx[c] = dy[c] / dx_column[c]

    def fitx(x, *p):
        val = 0.0
        for n, m in zip(p, x):
            val = val + n * m
        return val

    p0 = np.zeros(dx_column.size) + 0.1
    popt, pcov = curve_fit(
        fitx,
        dydx.T,
        B,
        p0=p0,
        #sigma=(np.arange(0, 360, 1) - 270) ** 2 * 10 + 1,
        #absolute_sigma=True,
    )
    #def errfunc2(p, x, y):
    #    """Error function defined as the difference between f(x) and data"""
    #    return (fitfunc2(p,x)-y)
#
    #def fitfunc2(p, x):
    #    """generates sum(p[i]*x[i])"""
    #    return fitx(x,*p)
    #    #val=0.0*x[0]
    #    #print(x.shape)
    #    #for n in range(len(p)):
    #    #    val=val+p[n]*x[n]
    #    #return val
    #
    #p1, success = leastsq(errfunc2,
    #              p0[:],
    #              args=(dydx.T, B))
    #print(p1)
    pstd = np.sqrt(np.diag(pcov))
    
    return popt, pstd, dydx


def rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2))
