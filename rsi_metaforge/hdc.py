from __future__ import annotations

import random
import math
from typing import List, Tuple, Dict

SEED = 42

class HDCSpace:
    """Hyperdimensional Computing / Vector Symbolic Architecture (VSA) space.
    Uses bipolar vectors (+1, -1) of dimension D.
    """
    def __init__(self, d: int = 10000, seed: int = SEED):
        self.d = d
        self.rng = random.Random(seed)
        self.vocab = {}

    def random_vector(self) -> List[int]:
        return [self.rng.choice([-1, 1]) for _ in range(self.d)]

    def get_vector(self, name: str) -> List[int]:
        if name not in self.vocab:
            self.vocab[name] = self.random_vector()
        return self.vocab[name]

    def bind(self, x: List[int], y: List[int]) -> List[int]:
        return [a * b for a, b in zip(x, y)]

    def bundle(self, vectors: List[List[int]]) -> List[int]:
        sums = [0] * self.d
        for v in vectors:
            for i in range(self.d):
                sums[i] += v[i]
        return [1 if s >= 0 else -1 for s in sums]

    def permute(self, x: List[int], shift: int = 1) -> List[int]:
        shift = shift % self.d
        return x[shift:] + x[:shift]

    def similarity(self, x: List[int], y: List[int]) -> float:
        dot = sum(a * b for a, b in zip(x, y))
        return dot / self.d


class KuramotoSolver:
    """Coupled Kuramoto Oscillator Network for program synthesis.
    Positions compete locally, and align globally with VSA target attractors.
    """
    def __init__(self, num_slots: int, vocab_size: int, seed: int = SEED):
        self.L = num_slots
        self.M = vocab_size
        self.rng = random.Random(seed)

    def solve(self, target_similarities: List[List[float]], steps: int = 150, dt: float = 0.05, 
              K: float = 2.0, J: float = 1.0, noise_std: float = 0.2) -> List[int]:
        phases = [[self.rng.uniform(0, 2 * math.pi) for _ in range(self.M)] for _ in range(self.L)]
        
        for _ in range(steps):
            new_phases = [[0.0] * self.M for _ in range(self.L)]
            for l in range(self.L):
                for j in range(self.M):
                    repulsion = 0.0
                    for k in range(self.M):
                        if k != j:
                            repulsion += math.sin(phases[l][j] - phases[l][k])
                    
                    c_lj = target_similarities[l][j]
                    attraction = c_lj * math.cos(phases[l][j])
                    
                    noise = self.rng.normalvariate(0.0, noise_std) if noise_std > 0 else 0.0
                    
                    dtheta = 1.0 - J * repulsion + K * attraction + noise
                    new_phases[l][j] = (phases[l][j] + dtheta * dt) % (2 * math.pi)
            phases = new_phases
            
        best_program = []
        for l in range(self.L):
            best_op = -1
            best_val = -2.0
            for j in range(self.M):
                val = math.cos(phases[l][j])
                if val > best_val:
                    best_val = val
                    best_op = j
            best_program.append(best_op)
        return best_program


class HDCAnalogicalTransfer:
    """Handles algebraic analogy queries using VSA.
    Translates structures: A is to B as C is to D => D = C * A^-1 * B
    """
    def __init__(self, hdc: HDCSpace):
        self.hdc = hdc

    def transfer(self, input_a: List[int], prog_a: List[int], input_b: List[int]) -> List[int]:
        # T = prog_a * input_a
        relation = self.hdc.bind(prog_a, input_a)
        # prog_b = input_b * relation
        prog_b = self.hdc.bind(input_b, relation)
        return prog_b
