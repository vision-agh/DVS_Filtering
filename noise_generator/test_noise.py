#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

import noise_generator_py as ng

width = 640
height = 480

# Create sample events (polarity 0 or 1)
events = [ng.Event2d(x=i % width, y=i % height, p=i % 2, t=i * 10) for i in range(100)]
original_set = {(e.x, e.y, e.p, e.t) for e in events}

print(f"Input events: {len(events)}")

# Create noise generator
noise_gen = ng.NoiseGeneratorAlgorithm(
    width=width, height=height,
    shot_noise_rate_hz=0.5, poisson_divider=20, timestamp_resolution_us=1
)

# Process and sort by timestamp so noise is embedded within the original stream
noisy_events = sorted(noise_gen.process_events(events), key=lambda e: e.t)
noise_count = sum(1 for e in noisy_events if (e.x, e.y, e.p, e.t) not in original_set)

print(f"Output events: {len(noisy_events)}  (original: {len(events)}, noise: {noise_count})")

# Print first 15 events showing which are noise
print("\nFirst 15 events (* = noise):")
for i, e in enumerate(noisy_events[:15]):
    tag = "*" if (e.x, e.y, e.p, e.t) not in original_set else " "
    print(f"  [{tag}] t={e.t:5d}  x={e.x:4d}  y={e.y:4d}  p={e.p}")
