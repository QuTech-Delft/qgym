/** \file
 * Defines miscellaneous API functions.
 */

#include <iostream>
#include <ql/ir/cqasm/read.h>
#include <ql/ir/cqasm/write.h>
#include <ql/ir/new_to_old.h>
#include <ql/ir/old_to_new.h>
#include <ql/api/api.h>
#include <ql/utils/num.h>
#include <ql/utils/filesystem.h>

#include "rl_ql/mapping.h"


//============================================================================//
//                               W A R N I N G                                //
//----------------------------------------------------------------------------//
//         Docstrings in this file must manually be kept in sync with         //
//     mapping.i! This should be automated at some point, but isn't yet.      //
//============================================================================//

namespace rl_ql {


void map_program(std::string input_program, std::string output_program, std::vector<int> mapping) {
    // construct an empty new-style ir
    auto ir = ql::utils::make<ql::ir::Root>();
    ql::ir::cqasm::ReadOptions read_options;
    read_options.load_platform = true;

    // read the input_program file into the ir
    ql::ir::cqasm::read_file(ir, input_program, read_options);

    // perform the mapping inside an ir conversion sandwich
    auto old_ir = ql::ir::convert_new_to_old(ir); // ir sandwich start
    for (auto &kernel: old_ir->kernels) {
        for (auto &gate: kernel->gates) {
            std::cout << gate->operands << std::endl;
            for (auto &vqubit: gate->operands) {
                vqubit = (ql::utils::UInt)mapping[vqubit];
            }
            std::cout << gate->operands << std::endl;
        }
    }
    auto new_ir = ql::ir::convert_old_to_new(old_ir); // ir sandwich end

    // write the new-style ir to the output_file
    ql::ir::cqasm::WriteOptions write_options;
    write_options.include_platform = true;
    ql::utils::OutFile file{output_program};
    ql::ir::cqasm::write(new_ir, write_options, file.unwrap());
}

} // namespace rl_ql
