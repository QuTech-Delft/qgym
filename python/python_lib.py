# This file is copied from OpenQL (Apache 2.0 License, Copyright [2016] [Nader Khammassi & Imran Ashraf, QuTech, TU Delft]): https://github.com/QuTech-Delft/OpenQL/blob/develop/LICENSE
# For the original file see: https://github.com/QuTech-Delft/OpenQL/blob/develop/python/python_lib.py
# No changes were made (apart from formatting according to the black code style).

import os
import sys
import sysconfig

name = "python" + "".join(map(str, sys.version_info[0:2]))
debug = sys.argv[1].lower() == "debug"

libdir = os.path.join(os.path.dirname(sysconfig.get_paths()["include"]), "libs")

options = []
for entry in os.listdir(libdir):
    entry = entry.lower()
    if not entry.startswith(name):
        continue
    if not entry.endswith(".lib"):
        continue
    s = entry.split("_", maxsplit=1)
    flags = s[1] if len(s) > 1 else ""
    entry_debug = "d" in flags
    if debug != entry_debug:
        print(
            "note: %s was not considered as Python library due to debug flag mismatch"
            % entry,
            file=sys.stderr,
        )
        continue
    options.append(entry)

if not options:
    print("PYTHON_LIBRARIES_NOTFOUND")
else:
    # if there are multiple, no idea how to choose which.
    options.sort()
    print(os.path.join(libdir, options[0]).replace("\\", "/"))
