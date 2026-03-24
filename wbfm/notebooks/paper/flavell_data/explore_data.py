#!/usr/bin/env python3
"""Explore neuron IDs and categorization."""

import json
import numpy as np

with open("atanas_kim_2023-2022-06-28-07.json", "r") as f:
    data = json.load(f)

print("=== NEURON IDs ===")
print(f"Labeled neurons: {list(data['labeled'].keys())[:10]}...")
print(f"Total labeled: {len(data['labeled'])}")

print("\n=== NEURON CATEGORIZATION ===")
for cat, neurons in data["neuron_categorization"].items():
    print(f"Category {cat}: {len(neurons)} neurons")
    print(f"  Neurons: {neurons}")

print("\n=== SAMPLE NEURON LABELS ===")
for nid in list(data['labeled'].keys())[:5]:
    print(f"  Neuron {nid}: {data['labeled'][nid]}")
