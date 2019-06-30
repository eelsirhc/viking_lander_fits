#!/usr/bin/env python
import pandas as pd
import os, shutil
import subprocess
data =pd.read_csv("run_values",comment="#", index_col=0)

for irow, row in data.iterrows():
    print( "echo ",row.name)
    shutil.copytree("template",row.name, symlinks=True)
    namelist_file = os.path.join(row.name,"namelist.input")
    namelist = "".join(open(namelist_file).readlines())
    namelist = namelist.format(**row)

    output = open(namelist_file,'w')
    output.write(namelist)
    output.close()
##    print("(cd {0} ; ./ideal.single) &".format(row["name"]))
    print("(cd {0} ;qsub -N{0} runscript.wrf) &".format(row.name))
