// This file is largely based on OpenQL (Apache 2.0 License, Copyright [2016] [Nader Khammassi & Imran Ashraf, QuTech, TU Delft]): https://github.com/QuTech-Delft/OpenQL/blob/develop/LICENSE
// For the original file see: https://github.com/QuTech-Delft/OpenQL/blob/develop/python/openql.i
// Changes were made by renaming openql (and similar) to qgym, updating the docstring and removing unused statements.

%define DOCSTRING
"`QGym` internals for building a connection with OpenQL."
%enddef

%module(docstring=DOCSTRING) qgym
%feature("autodoc", "1");

%include "std_vector.i"
%include "std_string.i"

namespace std {
    %template(vectori) vector<int>;
}

%{
#include "qgym/api.h"
%}

// Include API features.
%include "qgym/mapping.i"
