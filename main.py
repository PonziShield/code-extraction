import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from bytecode_extraction.evm_bytecode_extraction import get_contract_bytecode, get_contract_name
from opcode_extraction.bytecode_disassembler.disassembler import Disassembler
from ast import literal_eval

if __name__ == '__main__':
    contract_addr = "0x815fF5e32F974862839C56f69CFD190F10E262F6"
    encoding_opcodes = []
    contract_name = get_contract_name(contract_addr)
    bytecode = get_contract_bytecode(contract_addr)

    disassembler_instance = Disassembler(bytecode)
    # opcode_sequences = disassembler_instance.get_disassembled_code()
    disassembler_instance.load_opcodes()
    opcodes = disassembler_instance.get_disassembled_opcode_values()


    # Encode each opcode as an integer
    for opcode in opcodes:
        opcode_hex = "0x" + opcode
        opcode_int = literal_eval(opcode_hex)
        encoding_opcodes.append(opcode_int)

    print(opcodes)
    print(encoding_opcodes)

