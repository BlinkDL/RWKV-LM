from __future__ import annotations

import random
import json
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .vm import (
    VMCrash, N_BASE_OPS, MACRO_ID_BASE, OP_NAMES, EXT_IMPL, EXT_TYPES,
    MAX_EXPANDED_LEN, MAX_STACK, run_base_program
)

# Constants
MAX_PROGRAM_LEN = 14
MACRO_DEPTH_CAP = 10
RESTARTS_PER_TASK = 6
ITERS_PER_RESTART = 550
DRIFT_RATE_INIT = 0.30
DRIFT_MIN, DRIFT_MAX = 0.10, 0.70
DRIFT_ETA = 0.25
NEG_GRAM_PROMOTE_AT = 25
NEG_GRAM_MAX_ACTIVE = 40

@dataclass(frozen=True)
class Macro:
    mid: int                       # token id (>= MACRO_ID_BASE)
    body: Tuple[int, ...]          # surface tokens; may contain earlier macros
    depth: int                     # 1 = base ops only; 2 = contains depth-1 macro
    wave_discovered: int
    parent_tasks: Tuple[str, ...]  # opaque task ids it was mined from

@dataclass
class SearcherState:
    macros: Dict[int, Macro] = field(default_factory=dict)
    weights: Dict[int, float] = field(default_factory=dict)
    version: int = 0
    history: List[str] = field(default_factory=list)
    extended: Dict[int, str] = field(default_factory=dict)

    def clone(self) -> "SearcherState":
        s = SearcherState()
        s.macros = dict(self.macros)
        s.weights = dict(self.weights)
        s.version = self.version
        s.history = list(self.history)
        s.extended = dict(self.extended)
        return s

    def vocab(self) -> List[int]:
        return (list(range(N_BASE_OPS)) + sorted(self.extended)
                + sorted(self.macros))

    def weight(self, tok: int) -> float:
        return self.weights.get(tok, 1.0)

    def digest(self) -> str:
        raw = json.dumps({
            "macros": {m.mid: list(m.body) for m in self.macros.values()},
            "weights": {k: round(v, 6) for k, v in sorted(self.weights.items())},
            "version": self.version,
        }, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

@dataclass(frozen=True)
class SealedTask:
    tid: str
    family: str
    train_pairs: Tuple[Tuple[Tuple[int, ...], int], ...]
    holdout_gate: Callable[[Callable[[List[int]], int]], bool]
    cf_gate: Callable[[Callable[[List[int]], int]], bool]

@dataclass
class SynthResult:
    tid: str
    solved: bool
    tokens: Tuple[int, ...]
    expanded_len: int
    evals_used: int
    best_score: float
    channel: str = "exploit"
    drift_evals: int = 0
    exploit_evals: int = 0
    neg_skips: int = 0
    zero_grams: Dict[Tuple[int, ...], int] = field(default_factory=dict)

def expand_tokens(tokens: Sequence[int], macros: Dict[int, Macro]) -> Tuple[int, ...]:
    out: List[int] = []

    def rec(seq: Sequence[int], depth: int) -> None:
        if depth > MACRO_DEPTH_CAP + 1:
            raise VMCrash("macro_depth_exceeded")
        for t in seq:
            if t < N_BASE_OPS or t in EXT_IMPL:
                out.append(t)
            elif t not in macros:
                raise VMCrash(f"unknown_token_{t}")
            else:
                m = macros.get(t)
                if m is None:
                    raise VMCrash("unknown_macro")
                rec(m.body, depth + 1)
            if len(out) > MAX_EXPANDED_LEN:
                raise VMCrash("program_too_long")

    rec(tokens, 1)
    return tuple(out)

def token_str(tokens: Sequence[int], macros: Dict[int, Macro]) -> str:
    parts = []
    for t in tokens:
        if t < N_BASE_OPS:
            parts.append(OP_NAMES[t])
        else:
            parts.append(f"M{t}")
    return " ".join(parts)

def _score_tokens(tokens: Sequence[int], macros: Dict[int, Macro],
                  pairs: Sequence[Tuple[Tuple[int, ...], int]]) -> float:
    try:
        expanded = expand_tokens(tokens, macros)
    except VMCrash:
        return 0.0
    screen = pairs[:6]
    s_hits = 0
    crashes_head = 0
    for k, (xs, y) in enumerate(screen):
        try:
            if run_base_program(expanded, xs) == y:
                s_hits += 1
        except VMCrash:
            if k < 2:
                crashes_head += 1
                if crashes_head == 2:
                    return 0.0
    if s_hits < len(screen):
        return 0.999 * (s_hits / len(pairs))
    hits = s_hits
    for xs, y in pairs[6:]:
        try:
            if run_base_program(expanded, xs) == y:
                hits += 1
        except VMCrash:
            pass
    return hits / len(pairs)

# Static type/arity model of the ISA
OP_TYPES: Dict[int, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    0: ((), ("i",)), 1: ((), ("i",)), 2: ((), ("i",)), 3: ((), ("i",)),
    4: ((), ("l",)),
    5: (("?",), ("?", "?")),          # DUP
    6: (("?", "?"), ("?", "?")),      # SWAP
    7: (("?",), ()),                  # DROP
    8: (("?", "?"), ("?", "?", "?")), # OVER
    9: (("i", "i"), ("i",)), 10: (("i", "i"), ("i",)), 11: (("i", "i"), ("i",)),
    12: (("i", "i"), ("i",)), 13: (("i", "i"), ("i",)),
    14: (("i", "i"), ("i",)), 15: (("i", "i"), ("i",)),
    16: (("l",), ("i",)), 17: (("l",), ("i",)), 18: (("l",), ("i",)),
    19: (("l", "i"), ("i",)),
    20: (("l",), ("l",)), 21: (("l",), ("l",)), 22: (("l",), ("l",)),
    23: (("l", "l"), ("l",)), 24: (("l", "l"), ("l",)),
    25: (("l", "l"), ("l",)), 26: (("l", "l"), ("l",)),
    27: (("l",), ("l",)), 28: (("l",), ("l",)),
    29: (("l",), ("i",)), 30: (("l",), ("i",)),
    31: (("l",), ("l",)), 32: (("l",), ("l",)),
    33: (("?", "?", "i"), ("?",)),    # SELECT
}

def _sim_types(seq: Sequence[int],
               sigs: Dict[int, Tuple[Tuple[str, ...], Tuple[str, ...]]],
               start: Tuple[str, ...] = ()) -> Optional[Tuple[str, ...]]:
    st: List[str] = list(start)
    for t in seq:
        sig = sigs.get(t)
        if sig is None:
            return None
        ins, outs = sig
        if len(st) < len(ins):
            return None
        popped = []
        for need in ins:
            v = st.pop()
            if need != "?" and v != "?" and v != need:
                return None
            popped.append(v)
        if t == 5:
            st.extend([popped[0], popped[0]])
        elif t == 6:
            st.extend([popped[0], popped[1]])
        elif t == 8:
            st.extend([popped[1], popped[0], popped[1]])
        elif t == 33:
            a, b = popped[2], popped[1]
            st.append(a if a == b else "?")
        else:
            st.extend(outs)
        if len(st) > MAX_STACK:
            return None
    return tuple(st)

def build_type_sigs(macros: Dict[int, Macro]) -> Dict[int, Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    sigs = dict(OP_TYPES)
    sigs.update(EXT_TYPES)
    for mid in sorted(macros):
        body = macros[mid].body
        found = None
        for k in range(0, 5):
            res = _sim_types(body, sigs, start=("?",) * k)
            if res is not None:
                ins = ("?",) * k
                found = (ins, tuple(res))
                break
        if found is None:
            sigs[mid] = (("?", "?", "?", "?"), ("?",))
        else:
            ins, full_out = found
            sigs[mid] = (ins, full_out)
    return sigs

def static_plausible(tokens: Sequence[int],
                     sigs: Dict[int, Tuple[Tuple[str, ...], Tuple[str, ...]]]) -> bool:
    res = _sim_types(tokens, sigs)
    return res is not None and len(res) == 1 and res[0] in ("i", "?")

def _sample_token(rng: random.Random, state: SearcherState) -> int:
    vocab = state.vocab()
    weights = [state.weight(t) for t in vocab]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for t, w in zip(vocab, weights):
        acc += w
        if r <= acc:
            return t
    return vocab[-1]

def _random_tokens(rng: random.Random, state: SearcherState, lo: int = 2,
                   hi: int = MAX_PROGRAM_LEN) -> List[int]:
    n = rng.randint(lo, hi)
    return [_sample_token(rng, state) for _ in range(n)]

def _mutate(rng: random.Random, state: SearcherState, tokens: List[int]) -> List[int]:
    t = list(tokens)
    op = rng.random()
    if op < 0.45 and t:
        t[rng.randrange(len(t))] = _sample_token(rng, state)
    elif op < 0.70 and len(t) < MAX_PROGRAM_LEN:
        t.insert(rng.randrange(len(t) + 1), _sample_token(rng, state))
    elif op < 0.90 and len(t) > 2:
        del t[rng.randrange(len(t))]
    else:
        if len(t) >= 2:
            i, j = rng.randrange(len(t)), rng.randrange(len(t))
            t[i], t[j] = t[j], t[i]
    return t

def _sample_token_drift(rng: random.Random, state: SearcherState) -> int:
    vocab = state.vocab()
    if rng.random() < 0.5:
        return vocab[rng.randrange(len(vocab))]
    ws = [state.weight(t) for t in vocab]
    mx = max(ws) if ws else 1.0
    inv = [(mx - w) + 0.25 for w in ws]
    total = sum(inv)
    r = rng.random() * total
    acc = 0.0
    for t, w in zip(vocab, inv):
        acc += w
        if r <= acc:
            return t
    return vocab[-1]

def _mutate_drift(rng: random.Random, state: SearcherState,
                  tokens: List[int]) -> List[int]:
    t = list(tokens)
    op = rng.random()
    if op < 0.40 and len(t) >= 2:
        i = rng.randrange(len(t))
        j = min(len(t), i + rng.randint(1, 3))
        t[i:j] = [_sample_token_drift(rng, state)
                  for _ in range(rng.randint(1, 3))]
    elif op < 0.70 and len(t) < MAX_PROGRAM_LEN - 1:
        i = rng.randrange(len(t) + 1)
        t[i:i] = [_sample_token_drift(rng, state)
                  for _ in range(rng.randint(1, 2))]
    else:
        for _ in range(rng.randint(2, 3)):
            if t:
                t[rng.randrange(len(t))] = _sample_token_drift(rng, state)
    return t[:MAX_PROGRAM_LEN]

def _grams(tokens: Sequence[int], lo: int = 2, hi: int = 3):
    n = len(tokens)
    for k in range(lo, hi + 1):
        for i in range(n - k + 1):
            yield tuple(tokens[i:i + k])

def _hits_negative(tokens: Sequence[int], active: set) -> bool:
    if not active:
        return False
    return any(g in active for g in _grams(tokens))

def synthesize(task_train_pairs: Sequence[Tuple[Tuple[int, ...], int]],
               tid: str,
               state: SearcherState,
               seed: int,
               restarts: int = RESTARTS_PER_TASK,
               iters: int = ITERS_PER_RESTART,
               seed_pool: Optional[Sequence[Tuple[int, ...]]] = None,
               drift_rate: float = DRIFT_RATE_INIT,
               neg_active: Optional[set] = None) -> SynthResult:
    evals = 0
    d_evals = 0
    e_evals = 0
    skips = 0
    zero_grams: Dict[Tuple[int, ...], int] = {}
    neg = neg_active or set()
    best_tokens: Tuple[int, ...] = ()
    best_score = -1.0
    best_channel = "exploit"
    sigs = build_type_sigs(state.macros)
    pool = [p for p in (seed_pool or []) if p]
    n_drift = max(0, min(restarts, round(restarts * drift_rate)))
    for r in range(restarts):
        is_drift = r < n_drift
        chan = "drift" if is_drift else "exploit"
        rng = random.Random((seed * 1_000_003) ^ (r * 7919) ^ (137 if is_drift else 0))
        if pool and not is_drift and r % 3 == 2:
            cur = _mutate(rng, state, list(rng.choice(pool)))
        elif is_drift:
            if pool and rng.random() < 0.8:
                cur = list(rng.choice(pool))
                for _ in range(rng.randint(2, 4)):
                    cur = _mutate_drift(rng, state, cur)
                if not cur:
                    cur = [_sample_token_drift(rng, state)]
            else:
                cur = [_sample_token_drift(rng, state)
                       for _ in range(rng.randint(3, MAX_PROGRAM_LEN))]
        else:
            cur = _random_tokens(rng, state)
        cur_score = _score_tokens(cur, state.macros, task_train_pairs)
        evals += 1
        if is_drift:
            d_evals += 1
        else:
            e_evals += 1
        if cur_score > best_score:
            best_score = cur_score
            best_tokens = tuple(cur)
            best_channel = chan
            if best_score >= 1.0:
                break
    try:
        expanded_len = len(expand_tokens(best_tokens, state.macros))
    except VMCrash:
        expanded_len = 0
    return SynthResult(
        tid=tid, solved=(best_score >= 1.0), tokens=best_tokens,
        expanded_len=expanded_len, evals_used=evals, best_score=best_score,
        channel=best_channel, drift_evals=d_evals, exploit_evals=e_evals,
        neg_skips=skips, zero_grams=zero_grams
    )
