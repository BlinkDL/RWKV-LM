from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Constants
MAX_EXPANDED_LEN = 20_000
MAX_STACK = 32
MAX_LIST_LEN = 256
INT_CAP = 10 ** 3000

class VMCrash(Exception):
    pass

OP_NAMES: Tuple[str, ...] = (
    "PUSH0", "PUSH1", "PUSH2", "PUSH3",            # 0..3
    "INPUT",                                        # 4
    "DUP", "SWAP", "DROP", "OVER",                  # 5..8
    "ADD", "SUB", "MUL", "DIVI", "MOD",             # 9..13
    "MAX2", "MIN2",                                 # 14..15
    "LEN", "HEAD", "LAST", "IDX",                   # 16..19
    "TAIL", "REVL", "SORTL",                        # 20..22
    "ZADD", "ZSUB", "ZMUL", "ZMAX",                 # 23..26
    "SCAN_MAX", "SCAN_ADD",                         # 27..28
    "RED_ADD", "RED_MAX",                           # 29..30
    "EVENIDX", "ODDIDX",                            # 31..32
    "SELECT",                                       # 33
)
N_BASE_OPS = len(OP_NAMES)
MACRO_ID_BASE = 100

def _need(stack: List[object], n: int) -> None:
    if len(stack) < n:
        raise VMCrash("stack_underflow")

def _pop_int(stack: List[object]) -> int:
    _need(stack, 1)
    v = stack.pop()
    if not isinstance(v, int):
        raise VMCrash("type_expected_int")
    return v

def _pop_list(stack: List[object]) -> Tuple[int, ...]:
    _need(stack, 1)
    v = stack.pop()
    if not isinstance(v, tuple):
        raise VMCrash("type_expected_list")
    return v

def _push(stack: List[object], v: object) -> None:
    if isinstance(v, int):
        if abs(v) > INT_CAP:
            raise VMCrash("int_overflow")
    elif isinstance(v, tuple):
        if len(v) > MAX_LIST_LEN:
            raise VMCrash("list_overflow")
    else:
        raise VMCrash("bad_value_type")
    if len(stack) >= MAX_STACK:
        raise VMCrash("stack_overflow")
    stack.append(v)

def _op_push_const(c):
    def f(stack, inp):
        _push(stack, c)
    return f

def _op_input(stack, inp):
    _push(stack, inp)

def _op_dup(stack, inp):
    _need(stack, 1); _push(stack, stack[-1])

def _op_swap(stack, inp):
    _need(stack, 2); stack[-1], stack[-2] = stack[-2], stack[-1]

def _op_drop(stack, inp):
    _need(stack, 1); stack.pop()

def _op_over(stack, inp):
    _need(stack, 2); _push(stack, stack[-2])

def _bin(fn):
    def f(stack, inp):
        b = _pop_int(stack); a = _pop_int(stack); _push(stack, fn(a, b))
    return f

def _lop(fn):
    def f(stack, inp):
        l = _pop_list(stack); _push(stack, fn(l))
    return f

def _zip2(fn):
    def f(stack, inp):
        b = _pop_list(stack); a = _pop_list(stack)
        n = min(len(a), len(b))
        _push(stack, tuple(fn(a[i], b[i]) for i in range(n)))
    return f

def _op_idx(stack, inp):
    i = _pop_int(stack); l = _pop_list(stack)
    _push(stack, l[i % len(l)] if l else 0)

def _scan_max(l):
    acc = []; cur = None
    for v in l:
        cur = v if cur is None else (cur if cur >= v else v)
        acc.append(cur)
    return tuple(acc)

def _scan_add(l):
    s = 0; acc = []
    for v in l:
        s += v; acc.append(s)
    return tuple(acc)

def _op_select(stack, inp):
    c = _pop_int(stack)
    ev = stack.pop() if stack else None
    tv = stack.pop() if stack else None
    if ev is None or tv is None:
        raise VMCrash("stack_underflow")
    _push(stack, tv if c != 0 else ev)

OP_IMPL = (
    _op_push_const(0), _op_push_const(1), _op_push_const(2), _op_push_const(3),
    _op_input,
    _op_dup, _op_swap, _op_drop, _op_over,
    _bin(lambda a, b: a + b), _bin(lambda a, b: a - b), _bin(lambda a, b: a * b),
    _bin(lambda a, b: 0 if b == 0 else a // b),
    _bin(lambda a, b: 0 if b == 0 else a % b),
    _bin(lambda a, b: a if a >= b else b), _bin(lambda a, b: a if a <= b else b),
    _lop(len), _lop(lambda l: l[0] if l else 0), _lop(lambda l: l[-1] if l else 0),
    _op_idx,
    _lop(lambda l: l[1:]), _lop(lambda l: tuple(reversed(l))),
    _lop(lambda l: tuple(sorted(l))),
    _zip2(lambda a, b: a + b), _zip2(lambda a, b: a - b),
    _zip2(lambda a, b: a * b), _zip2(lambda a, b: a if a >= b else b),
    _lop(_scan_max), _lop(_scan_add),
    _lop(sum), _lop(lambda l: max(l) if l else 0),
    _lop(lambda l: l[0::2]), _lop(lambda l: l[1::2]),
    _op_select,
)

# Extended registries
EXT_IMPL: Dict[int, Callable[[List[object], Tuple[int, ...]], None]] = {}
EXT_TYPES: Dict[int, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {}

def step_op(stack: List[object], op: int, inp: Tuple[int, ...]) -> None:
    if 0 <= op < N_BASE_OPS:
        OP_IMPL[op](stack, inp)
    elif op in EXT_IMPL:
        EXT_IMPL[op](stack, inp)
    else:
        raise VMCrash(f"unknown_op_{op}")

def run_base_program(expanded: Sequence[int], xs: Sequence[int]) -> int:
    if len(expanded) > MAX_EXPANDED_LEN:
        raise VMCrash("program_too_long")
    stack: List[object] = []
    inp = tuple(int(v) for v in xs)
    for op in expanded:
        step_op(stack, op, inp)
    if len(stack) != 1 or not isinstance(stack[0], int):
        raise VMCrash("bad_terminal_state")
    return stack[0]
