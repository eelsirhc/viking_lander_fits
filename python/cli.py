#!/usr/bin/env python
# Christopher Lee, Aeolis Research, 2019

import click
import numpy as np
import pandas as pd
import core
import os

from plotting import register as register_plots
from fitting import register as register_fits

@click.group()
def cli():
    pass



register_plots(cli)
register_fits(cli)

if __name__ == "__main__":
    cli()
