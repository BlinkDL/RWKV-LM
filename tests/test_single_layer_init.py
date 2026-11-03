"""Exercise CPU initializers without importing modules that compile CUDA on import."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "RWKV-v5/src/model.py",
    "RWKV-v5/rwkv_v6_demo.py",
    "RWKV-v7/train_temp/src/model.py",
    "RWKV-v7/train_temp/rwkv7_train_simplified.py",
)


def initializers():
    for filename in FILES:
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            assignments = [
                item for item in ast.walk(node)
                if isinstance(item, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "ratio_0_to_1" for t in item.targets)
            ]
            if assignments:
                yield filename, node, assignments[0].value


class SingleLayerInitializationTest(unittest.TestCase):
    def test_single_layer_initializers_are_finite(self):
        args = SimpleNamespace(
            n_layer=1, n_embd=64, dim_att=64, head_size=64,
            head_size_a=64, head_size_divisor=8, my_testing="",
        )
        for filename, node, _ in initializers():
            with self.subTest(file=filename, model=node.name):
                # Keep the real constructor; forward methods and CUDA imports are unnecessary here.
                node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"]
                namespace = {"torch": torch, "nn": nn, "MyModule": nn.Module, "math": math, "HEAD_SIZE": 64}
                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
                model = namespace[node.name](args, 0)
                for name, param in model.named_parameters():
                    self.assertTrue(torch.isfinite(param).all().item(), name)

    def test_single_layer_uses_maintainer_recommended_midpoint(self):
        for filename, node, value in initializers():
            with self.subTest(file=filename, model=node.name):
                ratio = eval(compile(ast.Expression(value), filename, "eval"), {
                    "args": SimpleNamespace(n_layer=1), "layer_id": 0,
                })
                self.assertEqual(ratio, 0.5)

    def test_multilayer_schedule_is_unchanged(self):
        for filename, node, value in initializers():
            expression = compile(ast.Expression(value), filename, "eval")
            for depth in (2, 3, 12, 24, 32):
                for layer in range(depth):
                    with self.subTest(file=filename, model=node.name, depth=depth, layer=layer):
                        ratio = eval(expression, {"args": SimpleNamespace(n_layer=depth), "layer_id": layer})
                        self.assertEqual(ratio, layer / (depth - 1))


if __name__ == "__main__":
    unittest.main()
