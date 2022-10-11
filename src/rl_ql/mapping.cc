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
            std::cout << gate->operands << std::endl; // ir sandwich end
        }
    }
    auto new_ir = ql::ir::convert_old_to_new(old_ir);

    // write the new-style ir to the output_file
    ql::ir::cqasm::WriteOptions write_options;
    write_options.include_platform = true;
    ql::utils::OutFile file{output_program};
    ql::ir::cqasm::write(new_ir, write_options, file.unwrap());
}

/**
 * Returns the compiler's version string.
 */
std::string get_version(std::string input_program, std::string output_program, std::vector<int> mapping) {

    for (auto map: mapping) {
        std::cout << "map entry: " << map << std::endl;
    }
    // create platform
    auto platf = ql::api::Platform("seven_qubits_chip", "cc_light");

    // create program
    auto prog = ql::api::Program("aProgram", platf, 2);

    // create kernel
    auto k = ql::api::Kernel("aKernel", platf, 2);

    k.cz(0, 1);
    k.measure(0);
    k.measure(1);

    prog.add_kernel(k);

    // add kernel to program
    auto k2 = ql::api::Kernel("aKernel2", platf, 2);

    k2.gate("x", 0);

    // add kernel to program
    prog.add_kernel(k2);

    auto compiler = prog.get_compiler();
    compiler.clear_passes();
    auto report_pass = compiler.append_pass("io.cqasm.Report");
    report_pass.set_option("with_platform", "yes");

    // compile the program
    prog.compile();

    auto ir = ql::utils::make<ql::ir::Root>();
    ql::ir::cqasm::ReadOptions read_options;
    read_options.load_platform = true;

    ql::ir::cqasm::read_file(ir, "aProgram.io_cqasm_report.cq", read_options);


    auto old_ir = ql::ir::convert_new_to_old(ir);
    for (auto &kernel: old_ir->kernels) {
        for (auto &gate: kernel->gates) {
            std::cout << gate->operands << std::endl;
            for (auto &vqubit: gate->operands) {
                vqubit = (ql::utils::UInt)mapping[vqubit];
            }
            std::cout << gate->operands << std::endl;
        }
    }

    auto new_ir = ql::ir::convert_old_to_new(old_ir);

    ql::ir::cqasm::WriteOptions write_options;
    write_options.include_platform = true;

    ql::utils::OutFile file{output_program};

    ql::ir::cqasm::write(new_ir, write_options, file.unwrap());

    std::cout << "Seems good to me! + test" << std::endl;
    return "WIP";
}
} // namespace ql
