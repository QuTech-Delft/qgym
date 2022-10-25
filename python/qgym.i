/**
 * @file   qgym.i
 * @author Imran Ashraf
 * @brief  swig interface file
 */
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
