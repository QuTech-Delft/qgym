import openql as ql
from qgym import map_program


def init():
    ql.initialize()  # init openql
    ql.set_option('output_dir', 'openql_output')  # set output directory for many (but not all stuff)
    ql.set_option('log_level', 'LOG_INFO')  # set log level


def make_one_kernel_program() -> ql.Program:
    # define the program and such
    platform = ql.Platform('my_platform', 'none')

    n_qubits = 3
    program = ql.Program('my_program', platform, n_qubits)
    kernel = ql.Kernel('my_kernel', platform, n_qubits)

    for qubit_idx in range(n_qubits):
        kernel.prepz(qubit_idx)

    kernel.x(0)
    kernel.hadamard(1)
    kernel.cz(2, 0)
    kernel.measure(0)
    kernel.measure(1)

    program.add_kernel(kernel)

    return program


def test_map_program_one_kernel():
    init()
    program = make_one_kernel_program()

    # set the compiler correctly
    compiler = program.get_compiler()
    compiler.clear_passes()
    report_pass = compiler.append_pass("io.cqasm.Report")
    report_pass.set_option("with_platform", "yes")
    report_pass.set_option("output_prefix", "openql_output/my_program")
    report_pass.set_option("output_suffix", ".qasm")

    # compile, ergo write the program
    program.compile()

    # map the program
    map_program("openql_output/my_program.qasm", "openql_output/my_program_mapped.qasm", [1, 2, 0])
