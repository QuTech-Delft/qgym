# This file is copied from OpenQL (Apache 2.0 License, Copyright [2016] [Nader Khammassi & Imran Ashraf, QuTech, TU Delft]): https://github.com/QuTech-Delft/OpenQL/blob/develop/LICENSE
# For the original file see: https://github.com/QuTech-Delft/OpenQL/blob/develop/python/compat/fix-swig-cmdline.py
# No changes were made (apart from formatting according to the black code style).

import sys
import os
import subprocess

swig = sys.argv[1]
args = []
for arg in sys.argv[2:]:
    if arg.startswith("--FIX,"):
        inc_dirs = arg[6:].split(",SEP,")
        for inc_dir in inc_dirs:
            args.append("-I" + inc_dir)
    else:
        args.append(arg)

cmdline = [swig] + args

if "VERBOSE" in os.environ:
    print("Fixed swig command line: " + " ".join(cmdline))

sys.exit(subprocess.run(cmdline).returncode)
