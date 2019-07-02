#!/usr/bin/env python
import pandas as pd
import os, shutil
import subprocess
data =pd.read_csv("run_values",comment="#", index_col=0)

runfile = "runfile"
cleanfile = "cleanfile"
ready=True
if os.path.exists(runfile):
	print("please delete {} before starting".format(runfile))
	ready=False
if os.path.exists(cleanfile):
	print("please delete {} before starting".format(cleanfile))
	ready=False

if not ready:
	import sys
	sys.exit(0)


runfile_output = open(runfile,'w')
for irow, row in data.iterrows():
    runfile_output.write("echo {}\n".format(row.name))
    if os.path.exists(row.name):
    	print("{} exists, skipping".format(row.name))
    	continue
    shutil.copytree("template",row.name, symlinks=True)
    namelist_file = os.path.join(row.name,"namelist.input")
    if os.path.exists(namelist_file):
    	namelist = "".join(open(namelist_file).readlines())
    	namelist = namelist.format(**row)
    	output = open(namelist_file,'w')
    	output.write(namelist)
    	output.close()
    else:
    	print("Namelist file not found in {}".format(row.name))
    runfile_output.write("(cd {0} ;qsub -N{0} runscript.wrf ; cd ..)\n".format(row.name))

runfile_output.close()

cleanfile_output = open(cleanfile,'w')
for irow, row in data.iterrows():
    cleanfile_output.write("echo {}\n".format(row.name))
    cleanfile_output.write("rm -rf {}\n".format(row.name))
cleanfile_output.write("rm -f {} {}\n".format(runfile, cleanfile))
cleanfile_output.close()


print("please run {}".format(runfile))