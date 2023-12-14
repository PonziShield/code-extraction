from enum import Enum
from typing import Optional
import math
from opcodes import Opcodes

class Opcode:
    def __init__(self, offset: int = 0, opcode: Opcodes = Opcodes.UNKNOWN, parameter: Optional[int] = None):
        self.offset = offset
        self.opcode = opcode
        self.parameter = parameter

    def get_opcode(self) -> Opcodes:
        return self.opcode

    def set_opcode(self, opcode: Opcodes):
        self.opcode = opcode

    def get_parameter(self) -> Optional[int]:
        return self.parameter

    def set_parameter(self, parameter: Optional[int]):
        self.parameter = parameter

    def get_offset(self) -> int:
        return self.offset

    def set_offset(self, offset: int):
        self.offset = offset

    def __str__(self) -> str:
        hex_offset = f"0x{self.offset:03X}"
        opcode_name = self.opcode.name
        parameter_str = f" 0x{self.parameter:X}" if self.parameter is not None else ""
        return f"{hex_offset} {opcode_name}{parameter_str}"
