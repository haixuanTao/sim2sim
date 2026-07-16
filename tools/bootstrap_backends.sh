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
#   Nexus         pip           INSUFFICIENT. The dimforge-nexus3d wheel has
#                               none of the APIs the Nexus rows are measured
#                               with: snap_rgb, snap_rgb_async, raytrace_frame,
#                               body_pose, set_rbd_steps_per_frame. Those rows
#                               came from locally patched kiss3d + nexus3d
#                               builds (kiss3d PR #397, feat/python-rt-render,
#                               dimforge/nexus PR #16). Not reproducible from
#                               PyPI at any version.
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
[[ "${1:-}" == "--probe-only" ]] && PROBE_ONLY=1

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
probe "$BENCH_VENV" nexus '
import nexus3d as nx
need = ("snap_rgb", "snap_rgb_async", "raytrace_frame", "body_pose", "set_rbd_steps_per_frame")
api = dir(nx.NexusViewer)
missing = [n for n in need if n not in api]
if missing:
    print(f"  {"nexus":22s} PARTIAL PyPI wheel; missing the benchmarked APIs: {", ".join(missing)}")
else:
    print(f"  {"nexus":22s} OK     patched build detected")'

for v in "$RT/isaac-venv/bin/python" ; do
  [[ -x "$v" ]] && printf '  %-22s OK\n' "isaac" || printf '  %-22s ABSENT (out-of-band; see header)\n' "isaac"
done
[[ -d "$RT/Genesis/ext/LuisaRender/build/bin" ]] \
  && printf '  %-22s OK\n' "luisarender" \
  || printf '  %-22s ABSENT (source build; see header)\n' "luisarender"

cat <<'EOF'

Reproducible today: MuJoCo, Genesis+Nyx.
Not reproducible:   mjlab (upstream pin bug), Nexus (PyPI wheel lacks the
                    measured APIs), Isaac + LuisaRender (not automated here).
Rows measured with a backend marked BLOCKED/PARTIAL/ABSENT cannot be
re-measured on this machine -- do not "refresh" them without rebuilding the
toolchain first, or you will be comparing against numbers you cannot verify.
EOF
