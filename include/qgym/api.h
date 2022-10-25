/** \file
 * Main header for the external API to QGym.
 */

#pragma once

//============================================================================//
//                               W A R N I N G                                //
//----------------------------------------------------------------------------//
//  Additions to/removals from the API (classes & global functions) must be   //
//           manually kept in sync with the __all__ declaration in            //
//                          python/qgym/__init__.py                           //
//----------------------------------------------------------------------------//
//  Additions to/removals from the API fileset must manually be kept in sync  //
//  with the python/qgym.i (don't forget to add a .i subfile in src as well)  //
//----------------------------------------------------------------------------//
//    Additions to/removals from the set of automatically-wrapped or SWIG     //
//  STL template expansion types that the API uses must be kept in sync with  //
//    the C++ to Python typemap in the docstring monkey-patching logic of     //
//                          python/qgym/__init__.py                           //
//============================================================================//

#include "qgym/mapping.h"
