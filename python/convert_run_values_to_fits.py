#!/usr/bin/env python
from __future__ import print_function
import asciitable

data = asciitable.read("run_values", numpy=False)
data2 = asciitable.read("run_values")
low = []
high = []
names = list(data2.dtype.names)
print(names)
names[0] = "filename"
ibase = data["name"].index("base")
for r in data:
    r[1] = format(float(r[1]) / float(data[ibase][1]),"f")
    r[0] = "fit/{0}.fit".format(r[0])
    if r[0].count("base"):
        low.append(r)
        high.append(r)
    if r[0].count("low"):
        low.append(r)
    if r[0].count("high"):
        high.append(r)

asciitable.write(low, open("fit_low.parameters",'w'), names=names, delimiter=',')
asciitable.write(high, open("fit_high.parameters",'w'), names=names, delimiter=',')
