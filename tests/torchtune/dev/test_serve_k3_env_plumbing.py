"""Guard the serve_k3.sh -> Ray worker env-var plumbing.

A tunable that the launcher exports but that never reaches the Ray worker
process produces a *clean* run at fallback speed. That is indistinguishable
from a genuine null result, and it has already invalidated one campaign:
`VLLM_KIMI_XPU_KDA_TRITON` was exported at the top of serve_k3.sh but carried
by neither the per-node ssh export block nor
`VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`, so job 8744445's "Triton doesn't help"
measured the untouched Python fallback.

There are exactly two channels a var must travel, and a var needs BOTH:

1. the ssh block, which sets up the environment of the remote shell that runs
   `ray start` (and hence of the raylet and everything it forks);
2. `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`, which vLLM's `get_env_vars_to_copy()`
   uses to push driver values into each actor's runtime env.

These tests read serve_k3.sh as text rather than executing it -- they must run
on a login node with no XPU, no Ray, and no allocation.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVE_K3 = REPO_ROOT / "experiments" / "kimi_k3_serving" / "serve_k3.sh"

# Every env var whose value changes which code path a worker takes. Adding a
# new gate to serve_k3.sh without adding it here is fine; adding it here
# without plumbing it is a test failure -- which is the point.
GATED_ENV_VARS = [
    "VLLM_KIMI_XPU_KDA_VECTORIZED",
    "VLLM_KIMI_XPU_CONV1D_VECTORIZED",
    "VLLM_KIMI_XPU_KDA_TRITON",
    "VLLM_KIMI_XPU_CAUSAL_CONV1D_TRITON",
    "VLLM_KIMI_XPU_KDA_CHUNKED",
    "VLLM_XPU_ENABLE_XPU_GRAPH",
    "VLLM_XPU_ALLOW_TRITON_SAMPLER",
    # Added after it was found unplumbed DURING an A/B that was about to
    # measure it (2026-08-11). vLLM's own `VLLM_` prefix rule would have
    # carried it into Ray actors, so the leg might have worked by accident --
    # but the launcher must not depend on an upstream default it does not
    # control, and a flag resolved at model-construction time must reach the
    # worker before the model is built.
    "VLLM_KIMI_FUSE_SHARED_EXPERT_AR",
    # Fused single-kernel KDA decode. Resolved at import time in kda.py, so it
    # must reach the worker process, not just the driver.
    "VLLM_KIMI_XPU_KDA_FUSED_DECODE",
]


@pytest.fixture(scope="module")
def script_text():
    assert SERVE_K3.is_file(), f"missing launcher: {SERVE_K3}"
    return SERVE_K3.read_text()


@pytest.fixture(scope="module")
def ssh_block(script_text):
    """The single `ssh ... ray start` line that seeds every remote node."""
    lines = [ln for ln in script_text.splitlines() if "ray start --address=" in ln]
    assert len(lines) == 1, (
        f"expected exactly one remote `ray start` ssh line, found {len(lines)}; "
        "the plumbing assertions below only inspect one line"
    )
    return lines[0]


@pytest.fixture(scope="module")
def ray_copy_list(script_text):
    """Names appended to VLLM_RAY_EXTRA_ENV_VARS_TO_COPY by the for-loop."""
    match = re.search(r"^for ray_env_name in (.+?); do$", script_text, re.M)
    assert match, "could not find the VLLM_RAY_EXTRA_ENV_VARS_TO_COPY for-loop"
    return match.group(1).split()


@pytest.mark.parametrize("var", GATED_ENV_VARS)
def test_gate_is_exported_with_a_default(var, script_text):
    """Unset and 0 must not be ambiguous -- always export an explicit value."""
    assert re.search(rf"^export {var}=\$\{{{var}:-", script_text, re.M), (
        f"{var} is not exported with a ${{VAR:-default}} at the top of "
        "serve_k3.sh, so a worker cannot distinguish 'unset' from 'off'"
    )


@pytest.mark.parametrize("var", GATED_ENV_VARS)
def test_gate_reaches_remote_nodes_via_ssh(var, ssh_block):
    assert f"{var}=" in ssh_block, (
        f"{var} is missing from the per-node ssh export block, so the remote "
        "raylet (and every worker it forks) never sees it"
    )


@pytest.mark.parametrize("var", GATED_ENV_VARS)
def test_gate_reaches_ray_actors_via_copy_list(var, ray_copy_list, script_text):
    """Require the explicit list even for vars upstream would copy anyway.

    `get_env_vars_to_copy()` (vllm/ray/ray_env.py) unions several sources, two
    of which would already cover some of these: names registered in
    `vllm.envs.environment_variables` (true of `VLLM_XPU_ENABLE_XPU_GRAPH`)
    and names matching the copied `VLLM_` prefix. We still demand the explicit
    entry, because both of those are properties of the vLLM checkout rather
    than of this launcher -- an upstream rebase that renames a var or narrows
    the prefix set would silently unplumb it. The explicit list costs nothing
    and is the only channel this repo controls.
    """
    assert var in ray_copy_list, (
        f"{var} is absent from VLLM_RAY_EXTRA_ENV_VARS_TO_COPY; the driver's "
        "value will not be injected into Ray actor runtime envs"
    )


def test_ssh_forwarded_vars_use_shell_quoted_values(ssh_block):
    """Every forwarded value must go through printf %q, not raw interpolation."""
    raw = re.findall(r"\b([A-Z0-9_]+)=\$([A-Za-z_][A-Za-z0-9_]*)\b", ssh_block)
    unquoted = [name for name, val in raw if not val.endswith("_q")]
    assert not unquoted, (
        "these ssh-forwarded vars interpolate an unquoted shell variable "
        f"(expected a printf '%q' _q alias): {sorted(set(unquoted))}"
    )


def test_vectorized_paths_default_on(script_text):
    """The 65.20 tok/s reference config had both on; defaulting off is a trap."""
    for var in ("VLLM_KIMI_XPU_KDA_VECTORIZED", "VLLM_KIMI_XPU_CONV1D_VECTORIZED"):
        assert re.search(rf"^export {var}=\$\{{{var}:-1\}}", script_text, re.M), (
            f"{var} must default to 1 -- with it off, KDA/conv1d decode falls "
            "back to a scalar per-token Python loop and every unqualified "
            "throughput number is measuring that instead"
        )


@pytest.mark.parametrize("var", GATED_ENV_VARS)
def test_gate_is_recorded_in_run_metadata(var, script_text):
    """RESULTS_DISCIPLINE: a flag under test is echoed by the run that used it."""
    assert re.search(rf'^echo "{var.lower()}=\${var}"', script_text, re.M), (
        f"{var} is not echoed into $LOG_DIR/metadata, so a past run's log "
        "cannot be checked for which path it actually took"
    )
