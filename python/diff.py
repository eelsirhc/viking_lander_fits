#!/usr/bin/env python
import asciitable as at
from fit_core import *
import sys

one = at.read(sys.argv[1], numpy=False)
two = at.read(sys.argv[2], numpy=False)

print one