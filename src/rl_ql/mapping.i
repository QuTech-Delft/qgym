%feature("docstring") get_version
"""
Returns the compiler's version string.

Parameters
----------
None

Returns
-------
str
    version number as a string
"""

%feature("docstring") map_program
"""
This method maps the program from the input_file to the output_file according to the provided mapping.
"""

%include "rl_ql/mapping.h"
