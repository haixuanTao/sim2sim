#!/usr/bin/env bash
# Rebuild the backend toolchains this repo benchmarks, and report honestly on
# the ones that cannot be rebuilt.
#
# Written after rebuilding this box from scratch (2026-07-16, RTX 5090). The
# headline finding: "pip install -e .[mjlab,genesis,nexus]" does NOT reproduce
# the benchmark. Two of six backends come up clean; the rest need a source
# build, an out-of-band installer, or patched forks that are not on PyPI. Each
# failure below is a real one hit on this machine, not a guess.
#
#   MuJoCo        pip           WORKS
#   Genesis+Nyx   pip (git)     WORKS -- needs genesis git-main, not the 1.2.2
#                               wheel (which lacks gs.qd_float), + torch cu128
#                               for sm_120, + gs-nyx-plugin imported AFTER
#                               gs.init().
#   mjlab         pip           BROKEN upstream. mjlab 1.3.0 requires
#                               mujoco-warp>=3.7.0.1 with no upper bound, but
#                               ls_parallel was removed in mujoco-warp 3.9.1, so
#                               a fresh resolve (3.10.0.2) dies on import.
#                               Pinning mujoco-warp<3.9.1 then fails on
#                               warp.context; warp-lang<1.15 does not fix it.
#                               No working combination found by pinning.
#                               This repo's [mjlab] extra also omits scipy.
#   Nexus         pip           INSUFFICIENT -- but see --nexus-src below, which
#                               DOES reproduce the rows. The dimforge-nexus3d
#                               wheel has none of the APIs the Nexus rows are
#                               measured with: snap_rgb, snap_rgb_async,
#                               raytrace_frame, body_pose. Building from source
#                               works, and needs FIVE repos on four non-default
#                               branches, wired by [patch.crates-io] path
#                               entries that assume they are siblings:
#
#                                 nexus      local/integration
#                                 kiss3d     feat/rt-transform-fastpath
#                                 khal       feat/cuda-graph-capture-fixes
#                                 vortx      rebase/nexus-0.4
#                                 cuda-oxide main   (NVlabs fork)
#
#                               Branch choice is not documented anywhere and was
#                               recovered by diffing exposed symbols/features:
#                               local/integration is the only nexus branch with
#                               the full API set (feat/python-rt-render has a
#                               subset); rt-transform-fastpath is the only kiss3d
#                               branch stacking all three fixes; vortx and khal
#                               must both carry the cuda-oxide feature or cargo
#                               will not resolve.
#
#                               khal's feat/cuda-graph-capture-fixes used to pin
#                               cuda-device at /home/xavier/cuda-oxide-src -- an
#                               absolute path into another developer's home, so
#                               it could not build anywhere else. Cargo reads
#                               path deps eagerly even when no feature selects
#                               them, so that broke `cargo metadata` for the
#                               whole graph, not just --features cuda-oxide.
#                               Fixed upstream in haixuanTao/khal#5 (now a
#                               pinned git rev). The rewrite below is kept only
#                               for local checkouts predating that fix.
#   LuisaRender   source        NOT AUTOMATED. Genesis ships zero LuisaRender
#                               files; enabling gs.renderers.RayTracer means
#                               compiling ext/LuisaRender (CMake/xmake + CUDA).
#                               Use a low job count -- -j24 OOMs a 30 GB box.
#   Isaac Sim     out-of-band   NOT AUTOMATED. pip isaacsim needs its own
#                               Python 3.10 venv (~10 GB) + a one-time ~2 min
#                               RTX shader-cache compile that looks like a hang.
#
# Usage:  tools/bootstrap_backends.sh [--probe-only]
#
# --probe-only skips installation and just reports what the current venvs can
# actually do, which is the useful part before trusting any number.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="${RT_BUILD_DIR:-$HOME/rt_build}"
NYX_VENV="$RT/nyx-venv"
BENCH_VENV="$RT/bench-venv"
PROBE_ONLY=0
NEXUS_SRC=0
case "${1:-}" in
  --probe-only) PROBE_ONLY=1 ;;
  --nexus-src)  NEXUS_SRC=1 ;;
esac

have() { command -v "$1" >/dev/null 2>&1; }
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

if ! have uv; then
  say "installing uv (needed: system python3.12 often ships without ensurepip,"
  echo "   so a stdlib venv comes up with no pip and 'python -m venv' is a dead end)"
  [[ $PROBE_ONLY -eq 1 ]] || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ $PROBE_ONLY -eq 0 ]]; then
  say "Genesis + Nyx  -> $NYX_VENV"
  uv venv "$NYX_VENV" --python 3.12
  # cu128: the RTX 5090 is compute capability 12.0 (sm_120); older wheels have
  # no kernels for it and fall back or fail.
  uv pip install --python "$NYX_VENV/bin/python" torch --torch-backend=cu128
  # git, not the wheel: released genesis-world 1.2.2 lacks the API Nyx needs.
  uv pip install --python "$NYX_VENV/bin/python" \
    "git+https://github.com/Genesis-Embodied-AI/Genesis.git" gs-nyx-plugin \
    pillow imageio imageio-ffmpeg

  say "MuJoCo (+ attempt mjlab, nexus) -> $BENCH_VENV"
  uv venv "$BENCH_VENV" --python 3.12
  # scipy is missing from the [mjlab] extra; add it explicitly.
  ( cd "$REPO" && uv pip install --python "$BENCH_VENV/bin/python" -e ".[mujoco,mjlab,nexus]" scipy )
fi

if [[ $NEXUS_SRC -eq 1 ]]; then
  say "Nexus from source -> $RT/{nexus,kiss3d,khal,vortx,cuda-oxide}"
  command -v cargo >/dev/null || { echo "  cargo missing: install rust first"; exit 1; }
  # Siblings under $RT, because nexus's [patch.crates-io] uses ../<repo> paths.
  clone_at() {  # clone_at <repo> <branch>
    local repo="$1" branch="$2" dir="$RT/$1"
    [[ -d "$dir/.git" ]] || git clone -q "https://github.com/haixuanTao/$repo.git" "$dir"
    git -C "$dir" fetch -q origin "$branch" && git -C "$dir" checkout -q "$branch"
    printf '  %-11s %s\n' "$repo" "$(git -C "$dir" branch --show-current)"
  }
  clone_at nexus      local/integration
  clone_at kiss3d     feat/rt-transform-fastpath
  clone_at khal       feat/cuda-graph-capture-fixes
  clone_at vortx      rebase/nexus-0.4
  clone_at cuda-oxide main

  # Legacy safety net: khal-std once pinned cuda-device at an absolute
  # /home/xavier path (fixed in haixuanTao/khal#5, now a pinned git rev, so a
  # fresh clone needs nothing here). Only rewrites a stale local checkout.
  khal_std="$RT/khal/crates/khal-std/Cargo.toml"
  if grep -q 'path = "/home/xavier/cuda-oxide-src' "$khal_std"; then
    sed -i 's|path = "/home/xavier/cuda-oxide-src/crates/cuda-device"|path = "../../../cuda-oxide/crates/cuda-device"|' "$khal_std"
    echo "  NOTE: stale khal checkout predating khal#5 -- repointed cuda-device locally"
  fi

  uv pip install --python "$BENCH_VENV/bin/python" maturin
  ( cd "$RT/nexus/crates/nexus_python3d" && "$BENCH_VENV/bin/maturin" build --release ) || exit 1
  uv pip install --python "$BENCH_VENV/bin/python" --reinstall \
    "$(ls -t "$RT"/nexus/target/wheels/dimforge_nexus3d-*.whl | head -1)"
fi

say "PROBE -- what can actually be reproduced on this box"
probe() {  # probe <venv> <label> <python-snippet>
  local py="$1/bin/python" label="$2" code="$3"
  if [[ ! -x "$py" ]]; then printf '  %-22s SKIP   (%s missing)\n' "$label" "$1"; return; fi
  "$py" - <<PY 2>/dev/null || printf '  %-22s BLOCKED\n' "$label"
$code
PY
}

probe "$BENCH_VENV" mujoco '
import mujoco; print(f"  {"mujoco":22s} OK     {mujoco.__version__}")'

probe "$NYX_VENV" genesis+nyx '
import genesis, importlib.metadata as m
print(f"  {"genesis+nyx":22s} OK     genesis {genesis.__version__} / gs-nyx-plugin {m.version("gs-nyx-plugin")}")'

# `import mjlab` succeeds even when mjlab is unusable -- it dies later, in
# Simulation(): sim.py:_should_use_cuda_graph() reads wp.context.runtime, and
# warp >=1.14 no longer exposes `context` as a module attribute. Probe that
# exact condition rather than the import, or this reports a false OK.
probe "$BENCH_VENV" mjlab '
import warp as wp, mjlab, importlib.metadata as m  # noqa: F401
if not hasattr(wp, "context"):
    print(f"  {"mjlab":22s} BROKEN warp-lang {m.version("warp-lang")} has no wp.context; "
          f"mjlab {m.version("mjlab")} Simulation() reads wp.context.runtime (sim.py:426)")
elif tuple(int(x) for x in m.version("mujoco-warp").split(".")[:3]) >= (3, 9, 1):
    print(f"  {"mjlab":22s} BROKEN mujoco-warp {m.version("mujoco-warp")} removed ls_parallel (>=3.9.1)")
else:
    print(f"  {"mjlab":22s} OK     mjlab {m.version("mjlab")} / mujoco-warp {m.version("mujoco-warp")}")'

# The wheel imports fine; it simply is not the build the Nexus rows were
# measured with. Report that as PARTIAL, not OK and not a hard failure.
# NOTE: set_rbd_steps_per_frame lives on the state object (nexus.rs), NOT on
# NexusViewer -- probing it against the viewer reports a false PARTIAL even on a
# correct source build.
probe "$BENCH_VENV" nexus '
import nexus3d as nx
missing = [n for n in ("snap_rgb", "snap_rgb_async", "snap_rgb_flush", "raytrace_frame", "body_pose")
           if n not in dir(nx.NexusViewer)]
if missing:
    print(f"  {"nexus":22s} PARTIAL PyPI wheel, not the benchmarked build; missing: {", ".join(missing)}")
    print(f"  {"":22s}         -> rebuild with: tools/bootstrap_backends.sh --nexus-src")
else:
    print(f"  {"nexus":22s} OK     source build (patched kiss3d/khal/vortx) detected")'

for v in "$RT/isaac-venv/bin/python" ; do
  [[ -x "$v" ]] && printf '  %-22s OK\n' "isaac" || printf '  %-22s ABSENT (out-of-band; see header)\n' "isaac"
done
[[ -d "$RT/Genesis/ext/LuisaRender/build/bin" ]] \
  && printf '  %-22s OK\n' "luisarender" \
  || printf '  %-22s ABSENT (source build; see header)\n' "luisarender"

cat <<'EOF'

Reproducible: MuJoCo, Genesis+Nyx, and Nexus via --nexus-src (verified: the
              source build runs cube raster + rt_native end to end).
Not:          mjlab (upstream pin bug -- no working combination by pinning),
              Isaac + LuisaRender (not automated here).
Rows measured with a backend marked BROKEN/PARTIAL/ABSENT cannot be re-measured
on this machine -- do not "refresh" them without rebuilding the toolchain first,
or you will be comparing against numbers you cannot verify.

Caveat on the Nexus source build: it reproduces the rows' pipeline but not their
numbers. On driver 580.159.03 it measures ~15% below the panel's driver-595
figures (cube raster 218 vs 271 gen-fps; rt_native 148 vs 175 fps, physics
3.62 vs 2.89 ms). Same box, different driver -- so treat the published Nexus
rows as driver-specific rather than reproducible constants.
EOF
