#!/bin/bash
cd ~/sim2sim
OUT=/tmp/claude-1000/-home-baguette/368b01ba-5a9b-4bcf-a861-1c5faf4fc743/scratchpad
for SIM in mujoco mjlab genesis nexus nexus_cuda_graph; do
  echo "=== $SIM $(date +%T) ==="
  MUJOCO_GL=egl timeout 3000 .venv/bin/python examples/lerobot_legs/batch_bench.py \
    --sim $SIM --envs 2048 --steps 200 > $OUT/batch2k_$SIM.log 2>&1
  echo "exit=$? $(grep -h '\[batch\]' $OUT/batch2k_$SIM.log)"
done
echo "=== done $(date +%T) ==="
