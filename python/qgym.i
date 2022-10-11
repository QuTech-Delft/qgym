/**
 * @file   qgym.i
 * @author Imran Ashraf
 * @brief  swig interface file
 */
%define DOCSTRING
"`OpenQL` is a C++/Python framework for high-level quantum programming. The framework provides a compiler for compiling and optimizing quantum code. The compiler produces the intermediate quantum assembly language in cQASM (Common QASM) and the compiled eQASM (executable QASM) for various target platforms. While the eQASM is platform-specific, the quantum assembly code (QASM) is hardware-agnostic and can be simulated on the QX simulator."
%enddef

%module(docstring=DOCSTRING) qgym
%feature("autodoc", "1");

%include "std_vector.i"
%include "std_string.i"

namespace std {
    %template(vectori) vector<int>;
}

%{
#include "rl_ql/api.h"
%}

// Include API features.
%include "rl_ql/mapping.i"
