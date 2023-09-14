from collections import deque, namedtuple

Kernel = namedtuple("Kernel", ["name", "repetitions", "gates", "cycles"])

# Sets are used for quick lookup
NOT_SUPPORTED = {
    "u ",
    "not",
    "display",
    "display_binary",
    "reset-averaging",
    "load_state",
}
ONE_QUBIT_INSTRUCTIONS = {
    "mx90",
    "x90",
    "x",
    "my90",
    "y90",
    "y",
    "z",
    "i",
    "h",
    "sdag",
    "s",
    "tdag",
    "t",
    "prep_x",
    "prep_y",
    "prep_z",
    "prep",
    "measure_x",
    "measure_y",
    "measure_z",
    "measure",
}
ONE_QUBIT_ONE_ANGLE_INSTRUCTIONS = {"rx", "ry", "rz"}
TWO_QUBIT_INSTRUCTIONS = {"cnot", "cz", "swap"}


class KernelParser:
    def __init__(self, name: str, repetitions: int) -> None:
        self.name = name
        self.repetitions = repetitions
        self.current_cycle = 0
        self.instructions = deque()
        self.cycles = deque()

    def add_instruction_line(self, instruction_line: str) -> None:
        if instruction_line.startswith("wait"):
            raise ValueError("The 'wait' instruction is not supported by this parser.")

        elif instruction_line.startswith("skip"):
            skip_count = int(instruction_line.removeprefix("skip"))
            self.current_cycle += skip_count
        else:
            for instruction in instruction_line.split("|"):
                self._parse_non_timing_instruction(instruction.strip())
                self.cycles.append(self.current_cycle)
            self.current_cycle += 1

    def _parse_non_timing_instruction(self, instruction: str) -> None:
        if instruction == "measure_all":
            self.instructions.append(["measure_all"])
            return

        instruction_name = instruction[: instruction.index(" ")]
        if instruction_name in NOT_SUPPORTED:
            raise ValueError(
                f"The '{instruction_name} instruction is currently not supported by this parser"
            )

        if instruction_name == "crk":
            q1, q2, k = instruction.removeprefix("crk").split(",")
            self.instructions.append(
                [instruction_name, qubit_to_int(q1), qubit_to_int(q2), int(k)]
            )

        elif instruction_name == ("cr"):
            q1, q2, angle = instruction.removeprefix("cr").split(",")
            self.instructions.append(
                ["cr", qubit_to_int(q1), qubit_to_int(q2), float(angle)]
            )

        elif instruction_name == "toffoli":
            q1, q2, q3 = instruction.removeprefix("toffoli").split(",")
            self.instructions.append(
                ["toffoli", qubit_to_int(q1), qubit_to_int(q2), qubit_to_int(q3)]
            )

        elif instruction_name == "measure_parity":
            q1, axis1, q2, axis2 = instruction.removeprefix(instruction_name).split(",")
            self.instructions.append(
                [
                    "toffoli",
                    qubit_to_int(q1),
                    axis1.strip(),
                    qubit_to_int(q2),
                    axis2.strip(),
                ]
            )

        elif instruction_name in ONE_QUBIT_INSTRUCTIONS:
            q1 = instruction.removeprefix(instruction_name)
            self.instructions.append([instruction_name, qubit_to_int(q1)])

        elif instruction_name in ONE_QUBIT_ONE_ANGLE_INSTRUCTIONS:
            q1, angle = instruction.removeprefix(instruction_name).split(",")
            self.instructions.append([instruction_name, qubit_to_int(q1), float(angle)])

        elif instruction_name in TWO_QUBIT_INSTRUCTIONS:
            q1, q2 = instruction.removeprefix(instruction_name).split(",")
            self.instructions.append(
                [instruction_name, qubit_to_int(q1), qubit_to_int(q2)]
            )

        else:
            raise ValueError("Instruction not recognized")

    def make_kernel(self) -> Kernel:
        Kernel(self.name, self.repetitions, list(self.instructions), list(self.cycles))


def qubit_to_int(qubit: str) -> int:
    return int(qubit.strip(" \t\v\n\r\fq[]"))
