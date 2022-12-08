%feature("docstring") map_program
"""
Map the program defined in the file given by input_program using the given mapping and write the result to the file named output_program.

:param input_program: Test String to location of the cQASM file describing the input program.
:param output_program: String to location of the cQASM file that should contain the output program.
:param mapping: Mapping to apply to the input program.
"""

%include "qgym/mapping.h"
