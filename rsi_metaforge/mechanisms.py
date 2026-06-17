from __future__ import annotations

import math
import random
import json
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .vm import (
    step_op, N_BASE_OPS, VMCrash, run_base_program, MAX_STACK
)
from .search import (
    Macro, SearcherState, SynthResult, expand_tokens, build_type_sigs,
    _score_tokens, MAX_PROGRAM_LEN, MAX_EXPANDED_LEN
)

# S3 World Model Sentinels & Constants
WM_MIN_OBS = 4
WM_ABSTAIN = object()      # sentinel: model declines to predict
WM_CRASH = object()        # sentinel: model predicts a VMCrash

WM_INT_FAMILY: Tuple[Tuple[str, Callable[[int, int], int]], ...] = (
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("divi0", lambda a, b: 0 if b == 0 else a // b),
    ("mod0", lambda a, b: 0 if b == 0 else a % b),
    ("max", lambda a, b: a if a >= b else b),
    ("min", lambda a, b: a if a <= b else b),
)
WM_INT_MAP = dict(WM_INT_FAMILY)

@dataclass
class WMOpModel:
    obs: List[Tuple[Tuple[object, ...], Tuple[int, ...], Optional[Tuple[object, ...]]]] = field(default_factory=list)
    survivors: List[Tuple[str, object]] = field(default_factory=list)
    memo: Dict[Tuple[Tuple[object, ...], Tuple[int, ...]], Optional[Tuple[object, ...]]] = field(default_factory=dict)
    crash_pre_lens: set = field(default_factory=set)
    fitted: bool = False

class OpSemanticsModel:
    """S3 World Model over the stack VM. Learns what each opcode does from
    observed stack transitions."""
    def __init__(self) -> None:
        self.ops: Dict[int, WMOpModel] = {op: WMOpModel() for op in range(N_BASE_OPS)}
        self.acts = 0
        self.predicted = 0
        self.abstained = 0

    def act(self, op: int, pre_stack: Sequence[object], inp: Sequence[int]) -> None:
        st = list(pre_stack)
        self.acts += 1
        try:
            step_op(st, op, tuple(int(v) for v in inp))
            post: Optional[Tuple[object, ...]] = tuple(st)
        except VMCrash:
            post = None
        self.observe(op, tuple(pre_stack), tuple(int(v) for v in inp), post)

    def observe(self, op: int, pre: Tuple[object, ...], inp: Tuple[int, ...],
                post: Optional[Tuple[object, ...]]) -> None:
        m = self.ops[op]
        m.obs.append((pre, inp, post))
        m.memo[(pre, inp)] = post
        if post is None:
            m.crash_pre_lens.add(len(pre))
        self._refit(op)

    @staticmethod
    def _enumerate_hypotheses() -> List[Tuple[str, object]]:
        hyps: List[Tuple[str, object]] = []
        for name, _ in WM_INT_FAMILY:
            hyps.append(("intbin", name))
        for k in range(0, 4):
            hyps.append(("const", k))
        hyps.append(("pushinput", None))
        for k in (1, 2, 3):
            if k == 1:
                plans = [(), (0,), (0, 0)]
            elif k == 2:
                plans = [(0, 1), (1, 0), (1, 0, 1), (0, 1, 0)]
            else:
                plans = [(0, 1, 2), (2, 1, 0)]
            for plan in plans:
                hyps.append(("shuffle", (k, plan)))
        return hyps

    @staticmethod
    def _hyp_predict(hyp: Tuple[str, object], pre: Tuple[object, ...], inp: Tuple[int, ...]) -> object:
        kind, param = hyp
        if kind == "const":
            return pre + (int(param),)
        if kind == "pushinput":
            return pre + (tuple(inp),)
        if kind == "intbin":
            if len(pre) < 2:
                return WM_CRASH
            b, a = pre[-1], pre[-2]
            if not isinstance(a, int) or not isinstance(b, int):
                return WM_CRASH
            fn = WM_INT_MAP[param]
            return pre[:-2] + (fn(a, b),)
        if kind == "shuffle":
            k, plan = param
            if len(pre) < k:
                return WM_CRASH
            popped = tuple(reversed(pre[len(pre) - k:]))
            return pre[:len(pre) - k] + tuple(popped[i] for i in plan)
        return WM_ABSTAIN

    def _refit(self, op: int) -> None:
        m = self.ops[op]
        if len(m.obs) < WM_MIN_OBS:
            m.fitted = False
            m.survivors = []
            return
        surv: List[Tuple[str, object]] = []
        for hyp in self._enumerate_hypotheses():
            ok = True
            for pre, inp, post in m.obs:
                want = post if post is not None else WM_CRASH
                got = self._hyp_predict(hyp, pre, inp)
                if got is WM_ABSTAIN or got != want:
                    if not (got is WM_CRASH and want is WM_CRASH):
                        ok = False
                        break
            if ok:
                surv.append(hyp)
        m.survivors = surv
        m.fitted = bool(surv)

    def predict_step(self, op: int, pre: Tuple[object, ...], inp: Tuple[int, ...]) -> object:
        m = self.ops[op]
        hit = m.memo.get((pre, inp), WM_ABSTAIN)
        if hit is not WM_ABSTAIN:
            self.predicted += 1
            return WM_CRASH if hit is None else hit
        if m.fitted:
            preds = set()
            raw = []
            for hyp in m.survivors:
                p = self._hyp_predict(hyp, pre, inp)
                raw.append(p)
                preds.add("CRASH" if p is WM_CRASH else p)
            if len(preds) == 1:
                self.predicted += 1
                return raw[0]
        self.abstained += 1
        return WM_ABSTAIN

    def predict_program_stack(self, expanded: Sequence[int], xs: Sequence[int]) -> object:
        if len(expanded) > MAX_EXPANDED_LEN:
            return WM_CRASH
        st: Tuple[object, ...] = ()
        inp = tuple(int(v) for v in xs)
        for op in expanded:
            nxt = self.predict_step(op, st, inp)
            if nxt is WM_ABSTAIN:
                return WM_ABSTAIN
            if nxt is WM_CRASH:
                return WM_CRASH
            st = nxt
        return st


# S1 PRM Constants & classes
PRM_N_FEATS = 8
PRM_PREFIX_INPUTS = 8
PRM_BEAM_WIDTH = 6
PRM_BEAM_MAX_LEN = 8

def prm_prefix_features(prefix: Sequence[int], macros: Dict[int, Macro],
                        pairs: Sequence[Tuple[Tuple[int, ...], int]],
                        wm: Optional[OpSemanticsModel] = None) -> Tuple[Tuple[float, ...], int]:
    try:
        exp = expand_tokens(tuple(prefix), macros)
    except VMCrash:
        return ((0.0,) * (PRM_N_FEATS - 1) + (1.0,), 0)
    sl = pairs[:PRM_PREFIX_INPUTS]
    n = max(1, len(sl))
    top_exact = top_int = depth1 = crash = near = 0.0
    runs = 0
    for xs, y in sl:
        st: Optional[List[object]] = []
        if wm is not None:
            pred = wm.predict_program_stack(exp, xs)
        else:
            pred = WM_ABSTAIN
        if pred is WM_ABSTAIN:
            st = []
            runs += 1
            try:
                for op in exp:
                    step_op(st, op, tuple(xs))
            except VMCrash:
                st = None
        elif pred is WM_CRASH:
            st = None
        else:
            st = list(pred)
        if st is None:
            crash += 1.0
            continue
        if len(st) == 1:
            depth1 += 1.0
        top = st[-1] if st else None
        if isinstance(top, int):
            top_int += 1.0
            if top == y:
                top_exact += 1.0
            else:
                near += 1.0 / (1.0 + math.log1p(abs(top - y)))
    has_macro = 1.0 if any(t >= MACRO_ID_BASE for t in prefix) else 0.0
    feats = (top_exact / n, top_int / n, near / n, depth1 / n, crash / n,
             min(1.0, len(prefix) / MAX_PROGRAM_LEN), has_macro, 1.0)
    return feats, runs

class StepScorer:
    """Averaged perceptron PRM for sequence step scoring."""
    def __init__(self, n_feats: int = PRM_N_FEATS) -> None:
        self.n_feats = n_feats
        self.w = [0.0] * n_feats
        self.wsum = [0.0] * n_feats
        self.updates = 0
        self.seen = 0

    def score(self, feats: Sequence[float]) -> float:
        if self.seen:
            return sum(ws * f for ws, f in zip(self.wsum, feats)) / self.seen
        return sum(w * f for w, f in zip(self.w, feats))

    def update(self, feats: Sequence[float], label: int) -> None:
        margin = label * sum(w * f for w, f in zip(self.w, feats))
        if margin <= 0.0:
            self.w = [w + label * f for w, f in zip(self.w, feats)]
            self.updates += 1
        self.wsum = [ws + w for ws, w in zip(self.wsum, self.w)]
        self.seen += 1


# S2 Debate Committee Simulation
@dataclass
class GDEpisode:
    task: str
    accepted_key: str
    consensus: str                 # full | majority | none
    queries: int
    acts: int
    committee: Dict[str, int] = field(default_factory=dict)

def run_debate_on_vm_programs(
    programs: List[Tuple[int, ...]],
    pairs: Sequence[Tuple[Tuple[int, ...], int]],
    queries_budget: int = 5
) -> GDEpisode:
    """S2 committee debate on candidates to select the optimal implementation."""
    committee = {"enum": len(programs)}
    if not programs:
        return GDEpisode("VM_task", "", "none", 0, 0, committee)
    
    # Run debate check
    # Filter programs that solve the training pairs
    valid_programs = []
    for prog in programs:
        correct = True
        for xs, y in pairs:
            try:
                if run_base_program(prog, xs) != y:
                    correct = False
                    break
            except VMCrash:
                correct = False
                break
        if correct:
            valid_programs.append(prog)
            
    if not valid_programs:
        return GDEpisode("VM_task", "", "none", 0, len(programs), committee)
        
    # Consensus logic
    # Group by similarity on extended inputs to find consensus classes
    if len(valid_programs) == 1:
        accepted_str = str(valid_programs[0])
        return GDEpisode("VM_task", accepted_str, "full", 0, len(programs), committee)
    else:
        # Majority consensus on shortest code
        shortest = sorted(valid_programs, key=len)[0]
        return GDEpisode("VM_task", str(shortest), "majority", 0, len(programs), committee)


# S4 Meta-learning config selection helper
def auto_meta_tune_drift(drift_rate: float, success_rate: float) -> float:
    """S4 Meta-learning control law updates drift rate depending on yield."""
    if success_rate > 0.5:
        # Lower drift, trust learned weights
        return max(DRIFT_MIN, drift_rate - 0.05)
    else:
        # Raise drift, explore anti-weighted tail
        return min(DRIFT_MAX, drift_rate + 0.05)

DRIFT_MIN = 0.10
DRIFT_MAX = 0.70
