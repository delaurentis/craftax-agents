"""craftax-agents: bring a Craftax agent to the browser. The playable ONNX format (FORMAT.md), converters for PyTorch / JAX
checkpoints, a checker that plays a model on the exact C engine, and the registry of agents playcraftax.com offers (models.json).

Paths below are the kit's own folders in a clone: weights/ (downloaded checkpoints) and out/ (exported models), both git-ignored.
"""
import os

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(KIT, "weights")
OUT = os.path.join(KIT, "out")
REGISTRY = os.path.join(KIT, "models.json")
