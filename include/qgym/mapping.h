/** \file
 * Defines miscellaneous API functions.
 */

#pragma once

#include <string>
#include <vector>

//============================================================================//
//                               W A R N I N G                                //
//----------------------------------------------------------------------------//
//         Docstrings in this file must manually be kept in sync with         //
//      mapping.i! This should be automated at some point, but isn't yet.     //
//============================================================================//

namespace qgym {

/**
 * Map the program defined in the file given by input_program using the given mapping and write the result to the file named output_program.
 */
void map_program(std::string input_program, std::string output_program, std::vector<int> mapping);

} // namespace qgym
