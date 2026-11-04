#!/usr/bin/env python3
"""Generate valid LZS and LZ5 LHA archives from real input files."""

import argparse
from array import array
from pathlib import Path
import struct


def crc16_arc(data):
    crc = 0
    for byte in data:
        crc = (crc >> 8) ^ CRC_TABLE[(crc ^ byte) & 0xFF]
    return crc


CRC_TABLE = []
for value in range(256):
    crc = value
    for _ in range(8):
        crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    CRC_TABLE.append(crc)


class LArcEncoder:
    def __init__(self, method):
        self.method = method
        self.packed = bytearray()
        self.reservoir = 0
        self.available_bits = 0
        self.flag_index = 0
        self.flag_bits = 0

    def bits(self, value, count):
        self.reservoir = (self.reservoir << count) | value
        self.available_bits += count
        while self.available_bits >= 8:
            self.packed.append((self.reservoir >> (self.available_bits - 8)) & 0xFF)
            self.available_bits -= 8
        self.reservoir &= (1 << self.available_bits) - 1

    def start_lz5_token(self, literal):
        if self.flag_bits == 0:
            self.flag_index = len(self.packed)
            self.packed.append(0)
        if literal:
            self.packed[self.flag_index] |= 1 << self.flag_bits
        self.flag_bits = (self.flag_bits + 1) & 7

    def literal(self, byte):
        if self.method == "-lzs-":
            self.bits(1, 1)
            self.bits(byte, 8)
        else:
            self.start_lz5_token(True)
            self.packed.append(byte)

    def match(self, source, length):
        if self.method == "-lzs-":
            self.bits(0, 1)
            self.bits((2_031 + source) & 2_047, 11)
            self.bits(length - 2, 4)
        else:
            self.start_lz5_token(False)
            position = (source - 18) & 4_095
            self.packed.append(position & 0xFF)
            self.packed.append(((position >> 4) & 0xF0) | (length - 3))

    def finish(self):
        if self.available_bits:
            self.packed.append((self.reservoir << (8 - self.available_bits)) & 0xFF)
            self.available_bits = 0
            self.reservoir = 0
        return self.packed


def tokens(data, window_size, maximum_length):
    table = array("I", [0]) * (1 << 20)
    position = 0
    size = len(data)
    while position + 2 < size:
        key = (data[position] << 16) | (data[position + 1] << 8) | data[position + 2]
        slot = ((key * 0x9E3779B1) >> 12) & 0xFFFFF
        candidate = table[slot] - 1
        length = 0
        if candidate >= 0 and position - candidate <= window_size:
            limit = min(maximum_length, size - position)
            while length < limit and data[candidate + length] == data[position + length]:
                length += 1
        if length >= 3:
            yield -1, candidate, length
            advance = length
        else:
            yield data[position], 0, 0
            advance = 1
        for index in range(position, min(position + advance, size - 2)):
            key = (data[index] << 16) | (data[index + 1] << 8) | data[index + 2]
            table[((key * 0x9E3779B1) >> 12) & 0xFFFFF] = index + 1
        position += advance
    while position < size:
        yield data[position], 0, 0
        position += 1


def encode(data, method):
    encoder = LArcEncoder(method)
    window_size = 2_048 if method == "-lzs-" else 4_096
    maximum_length = 17 if method == "-lzs-" else 18
    literals = 0
    matches = 0
    for literal, source, length in tokens(data, window_size, maximum_length):
        if literal >= 0:
            encoder.literal(literal)
            literals += 1
        else:
            encoder.match(source, length)
            matches += 1
    return encoder.finish(), literals, matches


def member(name, data, method, crc):
    packed, literals, matches = encode(data, method)
    encoded_name = name.encode("utf-8")
    if len(encoded_name) > 230 or len(data) > 0xFFFFFFFF or len(packed) > 0xFFFFFFFF:
        raise ValueError("level-0 member is too large")
    header = bytearray(2)
    header.extend(method.encode("ascii"))
    header.extend(struct.pack("<III", len(packed), len(data), 0))
    header.extend((0x20, 0, len(encoded_name)))
    header.extend(encoded_name)
    header.extend(struct.pack("<H", crc))
    header.append(ord("U"))
    header[0] = len(header) - 2
    header[1] = sum(header[2:]) & 0xFF
    print(method, name, len(data), len(packed), literals, matches, flush=True)
    return header + packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("name")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--size", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = [(path.name, path.read_bytes()[:args.size]) for path in args.inputs]
    checksums = [crc16_arc(data) for _, data in inputs]
    for method in ("-lzs-", "-lz5-"):
        archive = bytearray()
        for (name, data), crc in zip(inputs, checksums):
            archive.extend(member(name, data, method, crc))
        archive.append(0)
        destination = args.output / f"{args.name}-{method[1:-1]}.lzh"
        destination.write_bytes(archive)
        print(destination, flush=True)


if __name__ == "__main__":
    main()
