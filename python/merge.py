#!/usr/bin/env python
from __future__ import print_function
import asciitable
import sys

run_values = asciitable.read(sys.argv[1], numpy=False)
fitl = asciitable.read(sys.argv[2], numpy=False)
fith = asciitable.read(sys.argv[3], numpy=False)

base = run_values["name"].index("base")


for v in fitl["names"]:
    vi = fitl["names"].index(v)
    b=1.0
    if v=="massair_target":
        b=run_values[v][base]
    print(v, b*0.5*(fitl["newvalue"][vi]+fith["newvalue"][vi]))
