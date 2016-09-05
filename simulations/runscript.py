#!/usr/bin/env python
import asciitable
import os, shutil
import subprocess
data =asciitable.read("run_values")

for row in data:
    print "echo ",row["name"]
    d_row=dict()
    for n in row.dtype.names:
        d_row[n] = row[n]

    shutil.copytree("template",row["name"], symlinks=True)
    namelist_file = os.path.join(row["name"],"namelist.input")
    namelist = "".join(open(namelist_file).readlines())
    namelist = namelist.format(**d_row)
    output = open(namelist_file,'w')
    output.write(namelist)
    output.close()
    print("(cd {0} ; ./ideal.single) &".format(row["name"]))
    print("(cd {0} ; sleep 240 ; qsub runscript.wrf) &".format(row["name"]))
