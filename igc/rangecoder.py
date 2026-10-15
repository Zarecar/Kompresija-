"""Binarni aritmetički koder (PAQ stil, 32 bita, samo celobrojna aritmetika).

Verovatnoće su 12-bitne: p = P(bit=1) * 4096, 1 <= p <= 4095.
Flush upisuje 4 bajta, pa je tok SAMOOGRANIČAVAJUĆI: dekoder pročita tačno
onoliko bajtova koliko je enkoder upisao i ne zavisi od dužine fajla.
Ovo je bitno za pošteno merenje (vidi Goldman/Craig izazov iz 2001).
"""
from __future__ import annotations

MASK32 = 0xFFFFFFFF


class Encoder:
    __slots__ = ("x1", "x2", "out")

    def __init__(self) -> None:
        self.x1 = 0
        self.x2 = MASK32
        self.out = bytearray()

    def encode(self, bit: int, p: int) -> None:
        x1, x2 = self.x1, self.x2
        xmid = x1 + ((x2 - x1) >> 12) * p
        if bit:
            x2 = xmid
        else:
            x1 = xmid + 1
        while not ((x1 ^ x2) & 0xFF000000):
            self.out.append(x2 >> 24)
            x1 = (x1 << 8) & MASK32
            x2 = ((x2 << 8) & MASK32) | 0xFF
        self.x1, self.x2 = x1, x2

    def finish(self) -> bytes:
        self.out += self.x1.to_bytes(4, "big")
        return bytes(self.out)


class Decoder:
    __slots__ = ("x1", "x2", "x", "data", "pos")

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data
        self.x1 = 0
        self.x2 = MASK32
        self.x = int.from_bytes(data[pos:pos + 4], "big")
        if pos + 4 > len(data):
            raise ValueError("skraćen aritmetički tok")
        self.pos = pos + 4

    def decode(self, p: int) -> int:
        x1, x2, x = self.x1, self.x2, self.x
        xmid = x1 + ((x2 - x1) >> 12) * p
        if x <= xmid:
            bit = 1
            x2 = xmid
        else:
            bit = 0
            x1 = xmid + 1
        while not ((x1 ^ x2) & 0xFF000000):
            x1 = (x1 << 8) & MASK32
            x2 = ((x2 << 8) & MASK32) | 0xFF
            if self.pos >= len(self.data):
                raise ValueError("skraćen aritmetički tok")
            x = ((x << 8) & MASK32) | self.data[self.pos]
            self.pos += 1
        self.x1, self.x2, self.x = x1, x2, x
        return bit


# Adaptivni brojač verovatnoće: 16-bitna verovatnoća + broj viđenja (<= LIMIT).
# Brzina učenja ~1/(n+1.5) -> približno KT/Laplace procenjivač, pa je cena
# "učenja ničega" na nasumičnim podacima ~0.5*log2(n) bita po kontekstu.
LIMIT = 255
DT = [int(65536 / (n + 1.5)) for n in range(LIMIT + 1)]


def counter_p12(v: int) -> int:
    p = (v >> 8) >> 4
    return 1 if p < 1 else (4095 if p > 4095 else p)


def counter_update(v: int, bit: int) -> int:
    p = v >> 8
    n = v & 0xFF
    p += (((bit << 16) - p) * DT[n]) >> 16
    if n < LIMIT:
        n += 1
    return (p << 8) | n


COUNTER_INIT = (32768 << 8) | 0
