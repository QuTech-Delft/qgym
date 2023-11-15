from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Iterator

import qgym.io.utils as io_utils
from qgym.custom_types import Gate
from qgym.envs.scheduling import MachineProperties
from qgym.io.kernel_parser import Kernel, KernelParser


class Program:
    def __init__(self, qubits: int, kernels: list[Kernel]) -> None:
        self.qubits = qubits
        self.kernels = kernels


class InternalRepresentation:
    def __init__(
        self,
        version: str,
        platform_json: dict[str, Any],
        name: str,
        program: list[Program],
    ):
        version = version
        self.platform_json = platform_json
        self.name = name
        self.program = program


class QasmParser:
    """This QASM parser is used to parse QASM files retrieved from passen in OpenQL.

    It is not a general QASM parser.
    """

    def __init__(self, filename: str | Path):
        self.filename = Path(filename)

    def read_circuit(self) -> list[Gate]:
        ...

    def read_machine_properties(self) -> MachineProperties:
        ...

    def read_internal_representation(self) -> InternalRepresentation:
        with self.filename.open(encoding="utf-8") as qasm_file:
            lines = io_utils.glue_lines(qasm_file, "\\")
        lines = io_utils.split_lines(lines, ";")
        lines = io_utils.remove_one_line_comments(lines, "#")
        lines = io_utils.remove_block_comments(lines, "/*", "*/")
        lines = io_utils.strip_lines(lines)

        lines_iter = iter(lines)
        program = deque()
        for _ in enumerate(lines):
            try:
                line: str = next(lines_iter)
            except StopIteration:
                break

            if line.startswith("version"):
                version = self._parse_version(line)

            elif line.startswith("pragma @ql.platform("):
                platform_json = self._parse_platform(line, lines_iter)

            elif line.startswith("pragma @ql.name("):
                name = self._parse_name(line)

            else:
                program.append(line)

        return InternalRepresentation(
            version, platform_json, name, self._parse_program(program)
        )

    @staticmethod
    def _parse_platform(line: str, lines_iter: Iterator[str]) -> dict[str, Any]:
        json_string = line.removeprefix("pragma @ql.platform(")
        count = 1
        while count:
            line = next(lines_iter)
            count += line.count("(") - line.count(")")
            if count:
                json_string += line
            else:
                json_string += line.removesuffix(")")

        return json.loads(json_string)

    @staticmethod
    def _parse_name(line: str) -> str:
        name = line.removeprefix('pragma @ql.name("')
        return name.removesuffix('")')

    @staticmethod
    def _parse_version(line: str) -> str:
        version = line.removeprefix("version").strip()
        if version not in {"1.0", "1.1", "1.2"}:
            raise ValueError(f"Version {version} is not spported.")
        return version.strip()

    def _parse_program(self, program: deque[str]) -> Program:
        line: str = program.popleft(program)
        qubits = int(line.removeprefix("qubits "))

        kernels = deque()
    
        if line.startswith("."):
            name, repetitions = self._parse_kernel_name_and_repetitions(line)
        else:
            name = ""
            repetitions = 1
            program.appendleft(line)

        kernel_parser = KernelParser(name, repetitions)
        for line in program:
            if line.startswith("."):
                kernels.append(kernel_parser.make_kernel())
                name, repetitions = self._parse_kernel_name_and_repetitions(line)
                kernel_parser = KernelParser(name, repetitions)
            else:
                kernel_parser.add_instruction_line(line)

        kernels.append(kernel_parser.make_kernel())

        return Program(qubits, list(kernels))
    
    @staticmethod
    def _parse_kernel_name_and_repetitions(line: str) -> tuple[str, int]:
        line = line.removeprefix(".")
        if "(" in line:
            line = line.removesuffix(")")
            name, repetitions_string = line.split("(")
            repetitions = int(repetitions_string)
        else:
            name = line
            repetitions = 1
        return name, repetitions