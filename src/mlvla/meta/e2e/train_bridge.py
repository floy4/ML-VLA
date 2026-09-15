# src/mlvla/meta/e2e/train_bridge.py
"""End-to-end meta-net training: VLA flow-matching loss through the merged LoRA.

Per step: sample one train domain -> condition-homogeneous batch -> meta-net
forward on k cached evidence episodes (+view-dropout for film_dropout) ->
mean over k rows -> *RMS scales -> assemble jax tensors -> jit loss_and_grad ->
disassemble -> torch backward -> AdamW on meta-net only.

Import order (binding): jax/jnp and mlvla.meta.e2e.lora_mapping (whose
transitive imports are numpy-only) are safe at module top; every openpi-derived
module enters only via ``JAXBackend`` inside main(), which sets the derived env
and sys.path at its own import — nothing openpi-touching may be imported
before it.
"""
from __future__ import annotations

import os

# torch shares the GPU with JAX in this process, so preallocation must be off
# before jax is imported (jax_backend repeats this as a setdefault, but jax is
# already alive in this module's top-level import).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
import torch.utils.dlpack
import yaml

from mlvla.meta.e2e.lora_mapping import assemble, disassemble

VIEW_DIM = 7


def model_args_for(variant: str, module_shapes, hyper: dict,
                   pose_dim: int = VIEW_DIM) -> dict:
    """Constructor kwargs such that ``Cls(**model_args)`` reconstructs exactly.

    film/film_dropout add ``view_dim`` (absent from the yaml hypernet section);
    concat derives ``condition_dim = dino_dim + pose_dim`` (V4 takes
    condition_dim, not dino_dim). ``pose_dim`` comes from the config's
    pose_feature scheme (view7/zero: 7 = legacy width; vggt/raymap: 2048) and
    defaults to the legacy 7. Keys outside the target constructor's signature
    are dropped so no extra kwarg ever leaks into the checkpoint.
    """
    if variant in ("film", "film_dropout"):
        return {
            "dino_dim": int(hyper["dino_dim"]),
            "view_dim": VIEW_DIM,
            "module_shapes": module_shapes,
            "condition_hidden_dim": int(hyper["condition_hidden_dim"]),
            "hidden_dim": int(hyper["hidden_dim"]),
            "head_bottleneck": int(hyper.get("head_bottleneck", 32)),
            "dict_dim": int(hyper.get("dict_dim", 64)),
        }
    if variant == "concat":
        return {
            "condition_dim": int(hyper["dino_dim"]) + pose_dim,
            "module_shapes": module_shapes,
            "condition_hidden_dim": int(hyper["condition_hidden_dim"]),
            "hidden_dim": int(hyper["hidden_dim"]),
            "id_dim": int(hyper.get("id_dim", 256)),
            "trunk_layers": int(hyper.get("trunk_layers", 6)),
            "trunk_hidden": int(hyper.get("trunk_hidden", 1024)),
        }
    raise ValueError(f"unknown variant {variant!r}")


def build_model(variant: str, module_shapes, hyper: dict,
                pose_dim: int = VIEW_DIM) -> torch.nn.Module:
    """film/film_dropout -> SharedFiLM; concat -> V4 direct head.

    Constructs from :func:`model_args_for` — the same parameterization that is
    saved into checkpoints, so saving and building can never drift apart.
    """
    from mlvla.meta.hypernet import SharedFiLMABHyperNetwork, V4DirectABHyperNetwork
    args = model_args_for(variant, module_shapes, hyper, pose_dim=pose_dim)
    if variant in ("film", "film_dropout"):
        return SharedFiLMABHyperNetwork(**args)
    if variant == "concat":
        return V4DirectABHyperNetwork(**args)
    raise ValueError(f"unknown variant {variant!r}")


def _build_fast_path_perm(keys, sizes_a, sizes_b, shapes, mapping, jax_paths,
                          jax_sizes, site_shapes):
    """One-time permutation between the torch flat LoRA layout (all A then all
    B, ``keys`` order) and the jax flat pack layout (``jax_paths`` order —
    exactly what ``JAXBackend.grad_packed`` consumes).

    Recovered by a sentinel pass through the REAL :func:`assemble` (never by
    reimplementing its transposes/reshapes): module values carry global flat
    indices (+1 offset), so the assembled tensors spell out where every
    torch-flat element lands in the jax layout. float32 keeps integers exact
    only below 2**24 and the flat layout is ~46M elements, so each index is
    split across two passes (low 24 bits + high part). Two kinds of positions
    hold constant zeros in the slow path and must NOT map into the torch flat
    layout: untrained keys (e.g. lm_head under the v1-compat filter, sentinel
    -1 -> perm entry -(2**24+1)) and slot positions no site ever writes
    (assemble zero-initializes, and the +1 offset makes the untouched 0
    distinguishable from real index 0) — both come out as perm entry -1.

    Returns ``(perm_fwd, perm_bwd)`` int64 cpu arrays — ``perm_fwd[jax_pos] =
    torch_pos`` (or -1 for constant-zero slots) and the inverse
    ``perm_bwd[torch_pos] = jax_pos`` — or ``None`` if the trained entries do
    not form a bijection (duplicate slot writes would need assemble's
    last-write-wins / disassemble's grad-duplication semantics, which a
    permutation cannot mirror; the caller then falls back to the slow path).
    """
    low_bits = 1 << 24
    total_a = int(sum(sizes_a))
    total = total_a + int(sum(sizes_b))
    cum_a = np.concatenate([[0], np.cumsum(sizes_a)])
    cum_b = np.concatenate([[0], np.cumsum(sizes_b)])

    def sentinel_pass(transform):
        mod = {}
        for i, k in enumerate(keys):
            ia = np.arange(cum_a[i] + 1, cum_a[i] + sizes_a[i] + 1, dtype=np.int64)
            ib = np.arange(total_a + cum_b[i] + 1,
                           total_a + cum_b[i] + sizes_b[i] + 1, dtype=np.int64)
            mod[k] = {"A": transform(ia).reshape(shapes[k]["A"]).astype(np.float32),
                      "B": transform(ib).reshape(shapes[k]["B"]).astype(np.float32)}
        for k in mapping.keys:  # untrained keys: constant-zero slots
            if k not in mod:
                mod[k] = {f: np.full(site_shapes[k][f], -1.0, dtype=np.float32)
                          for f in ("A", "B")}
        return assemble(mapping, mod)

    assembled_lo = sentinel_pass(lambda v: v % low_bits)
    assembled_hi = sentinel_pass(lambda v: v // low_bits)
    perm_fwd = np.empty(int(sum(jax_sizes)), dtype=np.int64)
    off = 0
    for p, sz in zip(jax_paths, jax_sizes):
        entry = assembled_hi[p].ravel().astype(np.int64) * low_bits \
            + assembled_lo[p].ravel().astype(np.int64)
        perm_fwd[off:off + sz] = np.where(entry > 0, entry - 1, -1)
        off += sz
    valid = perm_fwd >= 0
    if not np.array_equal(np.sort(perm_fwd[valid]), np.arange(total)):
        return None
    perm_bwd = np.empty(total, dtype=np.int64)
    perm_bwd[perm_fwd[valid]] = np.flatnonzero(valid)
    # Build-time self-check (CPU, ~seconds): random values through the real
    # assemble() must equal the permutation gather exactly, so the fast path
    # is bit-equal to the slow path's data movement by construction.
    rng = np.random.default_rng(0)
    rand_ab = {k: {f: rng.standard_normal(shapes[k][f], dtype=np.float32)
                   for f in ("A", "B")} for k in keys}
    rand_ab.update({k: {f: np.zeros(site_shapes[k][f], dtype=np.float32)
                        for f in ("A", "B")}
                    for k in mapping.keys if k not in rand_ab})
    flat = np.concatenate([rand_ab[k]["A"].ravel() for k in keys]
                          + [rand_ab[k]["B"].ravel() for k in keys])
    packed = np.concatenate([np.asarray(assemble(mapping, rand_ab)[p]).ravel()
                             for p in jax_paths])
    padded = np.concatenate([flat, np.zeros(1, dtype=np.float32)])
    if not np.array_equal(packed, padded[np.where(valid, perm_fwd, total)]):
        return None
    return perm_fwd, perm_bwd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("film", "film_dropout", "concat"), required=True)
    parser.add_argument("--output", type=Path, required=True)   # /data2/.../e2e_<variant>/
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-steps-override", type=int, default=None,
                        help="Benchmark-only cap on the number of training-loop "
                             "iterations. total_steps (cosine anchor, warmup, "
                             "val/save cadence) is untouched, so a capped run is a "
                             "prefix of the full run. Also enables the per-step "
                             "loss jsonl dump under --output (rank 0).")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile() for meta-net")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=("default", "reduce-overhead", "max-autotune"),
                        help="torch.compile() mode (default: reduce-overhead)")
    parser.add_argument("--ddp", action="store_true",
                        help="Multi-process data parallel (torchrun): one GPU per "
                             "process, meta-net grads averaged via NCCL all-reduce. "
                             "Effective batch = batch_size * world_size.")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Warm-start the meta-net from an e2e checkpoint's "
                             "state_dict. Scales are also frozen from the checkpoint: "
                             "recomputing them over a changed train-domain set would "
                             "rescale every physical LoRA and perturb all converged "
                             "domains.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    rung = cfg["rung"]                      # e.g. "fullvw4"
    train_domains = cfg["train_domains"]    # 8 domains (with __de2a6ce7 suffix)
    # None (or missing task_prompt) = multitask: no _TARGET_TASK filter, prompts
    # come per-episode from meta/tasks.parquet (prompt_from_task=True in
    # get_data_config), so one domain loader covers all of the domain's tasks.
    task = cfg.get("task_prompt")
    tr = cfg["training"]
    total_steps = args.steps or (500 if args.smoke else int(tr["max_steps"]))
    # Benchmark-only loop cap (see --max-steps-override): every schedule-bearing
    # quantity keeps using total_steps.
    loop_steps = total_steps if args.max_steps_override is None \
        else min(total_steps, int(args.max_steps_override))
    bench = args.max_steps_override is not None
    batch_size, lr, wd = int(tr["batch_size"]), float(tr["lr"]), float(tr["weight_decay"])
    warmup, clip, val_every, save_every = (int(tr["warmup"]), float(tr["clip"]),
                                           int(tr["val_every"]), int(tr["save_every"]))
    view_dropout = 0.3 if args.variant == "film_dropout" else 0.0
    evidence_k = int(tr.get("evidence_k", 8))
    # pose_feature scheme: {scheme: view7|zero|vggt|raymap, cache: <npz>, layer: int}
    # (absent -> legacy view7, cond width 1024+7). Drives both the bank's
    # view() vectors and the concat condition_dim below; recorded into every
    # checkpoint so export/eval rebuild the identical cond input.
    pose_cfg = cfg.get("pose_feature")
    if pose_cfg is not None:
        from mlvla.meta.view_params import pose_dim_for_scheme
        pose_dim = pose_dim_for_scheme(pose_cfg.get("scheme", "view7"))
        if args.variant != "concat" and pose_dim != VIEW_DIM:
            raise SystemExit(
                f"pose_feature scheme {pose_cfg.get('scheme')!r} (pose_dim={pose_dim}) "
                f"is only wired for the concat variant, not {args.variant!r}")
    else:
        pose_dim = VIEW_DIM
    # Optional per-domain sampling weights (default 1.0) and pseudo-domains
    # (carved out of a parent dataset, e.g. lighting_L3__deep) whose TRAIN pool
    # is the bank's train split for that domain key instead of the full dataset.
    domain_weights = {d: float(w) for d, w in cfg.get("train_domain_weights", {}).items()}
    pool_from_bank = set(cfg.get("train_pool_from_bank", ()))
    unknown = (set(domain_weights) | pool_from_bank) - set(train_domains)
    if unknown:
        raise SystemExit(f"config references unknown train domains: {sorted(unknown)}")

    # ---- targets/scales (identical to v1 weight-regression protocol) ----
    from mlvla.meta.weights.targets import load_direct_targets, selected_rows
    repo_root = Path(__file__).resolve().parents[4]
    domains_yaml = yaml.safe_load((repo_root / "configs" / "domains.yaml").read_text())
    oracle_paths = {d: Path(domains_yaml[d]["canonical_npz"]) for d in train_domains}
    rows = selected_rows(repo_root / f"configs/selected_modules_{rung}.json",
                         int(cfg["weight_target"]["max_modules"]))
    # v1 compatibility: the v1 Concat column (v4.pt) trained on 456 modules —
    # the current fullvw4 row set adds two lm_head modules whose B dimension is
    # the 257k vocab, inflating V4's direct heads from 713M to ~4.9B params
    # (untrainable on one A40). lm_head lora also receives zero gradient (the
    # flow-matching loss never touches lm_head), so excluding it reproduces
    # v1's module set exactly (verified set-equal against v4.pt).
    exclude = tuple(cfg["weight_target"].get("exclude_key_substrings", ()))
    if exclude:
        n_before = len(rows)
        rows = [r for r in rows if not any(s in r["key"] for s in exclude)]
        print(f"module filter {exclude}: {n_before} -> {len(rows)} rows", flush=True)
    domain_targets, shapes, modules = load_direct_targets(oracle_paths, rows)
    from mlvla.meta.weights.normalization import compute_rms_scales, scales_to_dict
    scales = scales_to_dict(compute_rms_scales(domain_targets))
    init_ck = None
    if args.init_checkpoint is not None:
        init_ck = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        assert set(init_ck["scales"]) == set(scales), \
            "init checkpoint scale key set does not match this config's module rows"
        # The warm-started weights encode predictions in the checkpoint's scale
        # parameterization (pred * scale -> physical LoRA); adopting freshly
        # computed scales (whose RMS now includes the added train domain) would
        # silently rescale the physical LoRA of every converged domain at step 0.
        scales = init_ck["scales"]
        print(f"warm-start: scales frozen from {args.init_checkpoint}", flush=True)
    del domain_targets  # targets only feed scale computation; e2e never regresses weights
    keys = sorted(scales)
    # cuda:0 everywhere (never bare "cuda"): with >1 JAX device, XLA's
    # cross-device transfers change the process CUDA current device mid-run
    # (observed 0->1 after backend init, ->3 after first sharded batch), which
    # silently retargets bare-"cuda" placements onto another GPU.
    torch.cuda.set_device(0)
    scale_a = torch.tensor([scales[k]["A"] for k in keys], dtype=torch.float32, device="cuda:0")
    scale_b = torch.tensor([scales[k]["B"] for k in keys], dtype=torch.float32, device="cuda:0")

    # ---- multi-process data parallel (one JAX backend per process; the
    # single-process SPMD route is unusable on jax 0.5.3: sharded-batch
    # executables read uninitialized memory — NaN losses and call-to-call
    # nondeterminism with identical inputs) ----
    if args.ddp:
        import torch.distributed as dist
        from datetime import timedelta
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        # Staggered heavy init (MLVLA_DDP_STAGGER_S below) delays later ranks'
        # first collective past NCCL's default 10-min timeout; 60 min covers the
        # stagger when all ranks init at rank-0 pace, but host/IO contention
        # (or another job sharing the GPUs) can slow later ranks' init ~3x —
        # MLVLA_DDP_NCCL_TIMEOUT_MIN widens the budget for such launches.
        _nccl_to = float(os.environ.get("MLVLA_DDP_NCCL_TIMEOUT_MIN", "60"))
        dist.init_process_group("nccl", timeout=timedelta(minutes=_nccl_to))
        print(f"[ddp] world={world} rank={rank} effective_batch={batch_size * world}", flush=True)
        # One rank's startup (oracle-target staging, 3B-param model host load,
        # 16 eager dataset builds, XLA compile of the fused loss) transiently
        # takes ~O(100GB) host RAM; 4 concurrent startups OOM-killed a rank on
        # a 503GB host (SIGKILL mid-init). Stagger so spikes don't co-occur.
        _stagger = rank * float(os.environ.get("MLVLA_DDP_STAGGER_S", "420"))
        if _stagger > 0:
            print(f"[ddp] rank={rank} staggering heavy init by {_stagger:.0f}s", flush=True)
            time.sleep(_stagger)
    else:
        dist, rank, world = None, 0, 1
    is_main = rank == 0

    # ---- evidence ----
    from mlvla.meta.e2e.evidence import EvidenceBank
    feature_dir = Path(cfg["feature_cache_dir"])
    bank = EvidenceBank(feature_dir, train_domains, pose_feature=pose_cfg)
    if is_main:
        print(f"[pose] scheme={pose_cfg.get('scheme') if pose_cfg else 'view7'} "
              f"pose_dim={pose_dim} "
              f"cache={pose_cfg.get('cache') if pose_cfg else None}", flush=True)

    # ---- torch meta-net ----
    torch.manual_seed(0)
    model_args = model_args_for(args.variant, shapes, cfg["hypernet"], pose_dim=pose_dim)
    model = build_model(args.variant, shapes, cfg["hypernet"], pose_dim=pose_dim).to("cuda:0")
    if init_ck is not None:
        model.load_state_dict(init_ck["state_dict"])
        print(f"warm-start: loaded state_dict from {args.init_checkpoint} "
              f"({len(init_ck['state_dict'])} tensors)", flush=True)
    if args.compile:
        print(f"Compiling meta-net with torch.compile(mode={args.compile_mode!r})...", flush=True)
        model = torch.compile(model, mode=args.compile_mode)
        print("Compilation done.", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_at(step: int) -> float:
        if tr.get("lr_constant"):
            # Warm-start domain absorption: flat lr (optionally ramped over
            # `warmup` steps) instead of the fresh-run warmup+cosine.
            return lr * min(1.0, step / max(1, warmup))
        if step <= warmup:
            return lr * step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    # ---- jax backend + per-domain loaders ----
    # Loaders are constructed strictly sequentially, one domain at a time: each
    # domain_loader call rewrites process globals (env + _TARGET_TASK/
    # _EPISODE_FILTER) that the eager dataset constructor consumes, so no other
    # openpi/dataset code may run between two constructions.
    from mlvla.meta.e2e.jax_backend import JAXBackend
    backend = JAXBackend(template_npz=oracle_paths[train_domains[0]],
                         batch_size=batch_size)
    train_iters, val_iters = {}, {}
    for domain in train_domains:
        root = Path(domains_yaml[domain]["dataset_root"])
        val_ids = set(bank.split_episode_ids(domain, "val")) | set(bank.split_episode_ids(domain, "test"))
        # train = full domain pool (expert-training semantics); the 13 val/test
        # evidence episodes per domain serve evidence/val monitoring only and
        # are NOT excluded from the train pool. Pseudo-domains listed in
        # train_pool_from_bank instead train on their bank train split (their
        # cache blob is the carved-out subset). Per-rank seed: DDP ranks must
        # not consume identical batches.
        train_pool = (set(bank.split_episode_ids(domain, "train"))
                      if domain in pool_from_bank else None)
        if is_main:
            print(f"[loader] {domain}: train_pool={'full' if train_pool is None else len(train_pool)} "
                  f"val_pool={len(val_ids)}", flush=True)
        loader_seed = 42 + 1000 * rank
        train_iters[domain] = backend.domain_loader(root, task, episode_ids=train_pool,
                                                    batch_size=batch_size, seed=loader_seed)
        val_iters[domain] = backend.domain_loader(root, task, episode_ids=val_ids,
                                                  batch_size=batch_size, seed=loader_seed)

    # pre-assembled per-domain val evidence for export parity with v1
    val_evidence = {d: bank.domain_mean(d, "val").to("cuda:0") for d in train_domains}
    val_views = {d: bank.view(d).to("cuda:0").unsqueeze(0) for d in train_domains}

    # ---- host<->device transfer batching ----
    # 916 per-key .cpu()/.to("cuda:0") transfers per step dominated step time
    # (~2.7s/it vs ~1.1s budget); move one flattened buffer each way instead.
    sizes_a = [int(np.prod(shapes[k]["A"])) for k in keys]
    sizes_b = [int(np.prod(shapes[k]["B"])) for k in keys]
    # assemble() consumes every mapping key; keys absent from the trained set
    # (e.g. lm_head under the v1-compat filter) contribute zeros — dead targets
    # whose gradients are zero anyway.
    site_shapes: dict[str, dict[str, tuple]] = {}
    for site in backend.mapping.sites:
        site_shapes.setdefault(site.key, {})[site.factor] = site.canonical_shape
    extra_zero_ab = {k: {f: np.zeros(site_shapes[k][f], dtype=np.float32)
                         for f in ("A", "B")}
                     for k in backend.mapping.keys if k not in scales}

    def lora_values(a_list, b_list) -> dict:
        flat_a = torch.cat([t.reshape(-1) for t in a_list]).detach().cpu().numpy()
        flat_b = torch.cat([t.reshape(-1) for t in b_list]).detach().cpu().numpy()
        vals, ia, ib = {}, 0, 0
        for i, k in enumerate(keys):
            na, nb = sizes_a[i], sizes_b[i]
            vals[k] = {"A": flat_a[ia:ia + na].reshape(shapes[k]["A"]),
                       "B": flat_b[ib:ib + nb].reshape(shapes[k]["B"])}
            ia += na
            ib += nb
        vals.update(extra_zero_ab)
        return vals

    def lora_grads(gmod) -> tuple[list, list]:
        flat_ga = torch.from_numpy(np.concatenate([gmod[k]["A"].ravel() for k in keys]))
        flat_gb = torch.from_numpy(np.concatenate([gmod[k]["B"].ravel() for k in keys]))
        ga = [t.view(*shapes[k]["A"]) for t, k in zip(flat_ga.to("cuda:0").split(sizes_a), keys)]
        gb = [t.view(*shapes[k]["B"]) for t, k in zip(flat_gb.to("cuda:0").split(sizes_b), keys)]
        return ga, gb

    # ---- dlpack zero-copy fast path (opt-in via MLVLA_FAST_PATH=1; default
    # OFF. A/B-verified bit-exact for 300/300 steps, but the speed gate missed:
    # ~0.43 s/step saved = 14.6% at production autotune-on settings (2.96 ->
    # 2.53 s/step) and 7.8% under the deterministic autotune-off protocol
    # (4.97 -> 4.58) — under the 25% merge bar, so training arms keep the
    # original path for comparability unless explicitly opted in. "0"/"false"/
    # "off" -> the original per-tensor CPU round-trip path, kept verbatim
    # for A/B equivalence runs and rollback) ----
    # Slow path per step: 916-tensor cuda->cpu->numpy->jnp->(pack)->numpy->cuda.
    # Fast path: one cuda gather into the jax pack layout + dlpack share, and the
    # inverse gather back — pure data movement, bit-identical values.
    fast_path = os.environ.get("MLVLA_FAST_PATH", "0").strip().lower() not in ("0", "false", "off")
    perm_fwd_t = perm_bwd_t = zero_rows_t = None
    total_a_flat = int(sum(sizes_a))
    if fast_path:
        built = _build_fast_path_perm(keys, sizes_a, sizes_b, shapes, backend.mapping,
                                      backend.lora_paths, backend.lora_flat_sizes,
                                      site_shapes)
        if built is None:
            print("[fast-path] flat permutation failed the bijection/self-check; "
                  "falling back to the slow path", flush=True)
            fast_path = False
        else:
            perm_fwd_np, perm_bwd_np = built
            zero_rows_t = torch.from_numpy(perm_fwd_np < 0).to("cuda:0")
            perm_fwd_t = torch.from_numpy(
                np.where(perm_fwd_np >= 0, perm_fwd_np, 0).astype(np.int64)).to("cuda:0")
            perm_bwd_t = torch.from_numpy(perm_bwd_np).to("cuda:0")
            print(f"[fast-path] permutation built: {perm_fwd_t.numel()} jax-flat / "
                  f"{perm_bwd_t.numel()} torch-flat elements "
                  f"({int(zero_rows_t.sum())} constant-zero slots)", flush=True)
    # Fused all-reduce bucket for the DDP branch (lazily sized at first use,
    # when grads first exist; ~2.85GB f32 for the 713M-param V4 head).
    ddp_bucket = None
    ddp_layout = None

    history = []
    rng_gen = torch.Generator().manual_seed(1000 * rank)
    sample_w = torch.tensor([domain_weights.get(d, 1.0) for d in train_domains])
    if is_main:
        eff = {d: f"{sample_w[i].item() / sample_w.sum().item():.0%}"
               for i, d in enumerate(train_domains)}
        print(f"[sampling] effective domain shares: {eff}", flush=True)
    t0 = time.time()
    loss_log = None
    if bench and is_main:
        args.output.mkdir(parents=True, exist_ok=True)
        loss_log = open(args.output / "per_step_loss.jsonl", "w")
    for step in range(1, loop_steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        domain = train_domains[int(torch.multinomial(sample_w, 1, generator=rng_gen))]
        obs, act = next(train_iters[domain])

        x = bank.sample(domain, k=evidence_k, generator=rng_gen).to("cuda:0")
        if view_dropout > 0.0:
            keep = (torch.rand(x.shape[0], 1, 1, device="cuda:0") >= view_dropout).float()
            x = x * keep
        v = bank.view(domain).to("cuda:0").unsqueeze(0).expand(x.shape[0], -1)

        model.train()
        if args.variant == "concat":
            # pose vector (scheme-aware via bank.view, [pose_dim]) broadcast
            # onto each of the 4 frames (same semantics as
            # view_params.append_view_params) -> [k,4,dino+pose_dim]; V4 frame-means.
            cond = torch.cat([x, v.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)
            pred = model(cond)
        else:
            pred = model(x, v)
        # mean over k evidence rows -> one LoRA; *scales -> physical values.
        # Kept per-key (not one torch.stack): module shapes are heterogeneous
        # across keys, and each key's scale is a scalar so the chain rule is
        # elementwise-identical to the stacked formulation.
        a_phys = [pred[k]["A"].mean(0) * scale_a[i] for i, k in enumerate(keys)]
        b_phys = [pred[k]["B"].mean(0) * scale_b[i] for i, k in enumerate(keys)]
        key = jax.random.fold_in(jax.random.PRNGKey(0), step)
        if fast_path:
            # Forward: flat torch layout (all A then all B, `keys` order) ->
            # one cuda gather into the jax pack layout -> zero-copy dlpack
            # share. Same jitted grad_packed executable and identical input
            # values as the slow path (build-time self-checked), minus the 916
            # per-tensor host round trips and loss_and_grad's in-jax _pack.
            flat_ab = torch.cat([a.reshape(-1) for a in a_phys]
                                + [b.reshape(-1) for b in b_phys]).detach()
            flat_jax_t = flat_ab[perm_fwd_t]
            flat_jax_t[zero_rows_t] = 0.0  # untrained slots: slow path's zeros
            # torch (gather) and XLA may use different CUDA streams; make the
            # gather visible before the jax kernel reads the shared buffer.
            torch.cuda.synchronize()
            loss, gflat = backend.grad_packed(
                jax.dlpack.from_dlpack(flat_jax_t), key, obs, act)
            loss = float(loss)
            if not math.isfinite(loss):
                raise RuntimeError(f"step {step}: non-finite loss")
            # Backward: zero-copy back to torch, inverse gather into the torch
            # flat layout, split into per-key canonical views (same
            # shapes/split semantics as lora_grads on the slow path).
            jax.block_until_ready(gflat)
            gflat_torch = torch.utils.dlpack.from_dlpack(gflat)[perm_bwd_t]
            ga = [t.view(*shapes[k]["A"])
                  for t, k in zip(gflat_torch[:total_a_flat].split(sizes_a), keys)]
            gb = [t.view(*shapes[k]["B"])
                  for t, k in zip(gflat_torch[total_a_flat:].split(sizes_b), keys)]
        else:
            module_ab = lora_values(a_phys, b_phys)
            tensors = {p: jnp.asarray(t) for p, t in assemble(backend.mapping, module_ab).items()}
            loss, grads = backend.loss_and_grad(dict(tensors), key, obs, act)
            loss = float(loss)
            if not math.isfinite(loss):
                raise RuntimeError(f"step {step}: non-finite loss")
            gmod = disassemble(backend.mapping, {p: np.asarray(g) for p, g in grads.items()})
            ga, gb = lora_grads(gmod)
        if not all(torch.isfinite(g).all() for g in ga + gb):
            raise RuntimeError(f"step {step}: non-finite lora grads")
        if loss_log is not None:
            loss_log.write(json.dumps(
                {"step": step, "loss": loss, "t": round(time.time() - t0, 3)}) + "\n")
            loss_log.flush()

        # chain: backprop through the scale multiply. JAX grads ga/gb are w.r.t.
        # the PHYSICAL values (a_phys = pred_mean * scale), so the true target
        # is dL/dpred = ga * scale. The plain inner product proxy = <a_phys, ga>
        # yields exactly that via autograd (da_phys/dpred = scale); weighting ga
        # by the scale a second time — as the sketch/ruling-6 formula does —
        # over-weights each module by its own scale (numerically verified:
        # literal formula gives ga*scale^2/k, true chain rule is ga*scale/k).
        proxy = sum((a_phys[i] * ga[i]).sum() + (b_phys[i] * gb[i]).sum()
                    for i in range(len(keys)))
        optimizer.zero_grad(set_to_none=True)
        proxy.backward()
        if args.ddp and fast_path:
            # Fused all-reduce: assemble every grad into ONE pre-allocated flat
            # bucket, a single dist.all_reduce, then scatter back. Per-element
            # math identical to the per-parameter loop below (SUM then /world);
            # one NCCL collective instead of one per parameter. The layout holds
            # PARAMETER references and dereferences p.grad fresh each step:
            # zero_grad(set_to_none=True) makes every backward allocate new
            # grad tensors, so capturing p.grad itself at first use would read
            # and write step-1's stale (orphaned) tensors from step 2 on.
            with torch.no_grad():
                if ddp_bucket is None:
                    ddp_layout = [(p, p.grad.numel())
                                  for p in model.parameters() if p.grad is not None]
                    ddp_bucket = torch.zeros(sum(n for _, n in ddp_layout),
                                             dtype=torch.float32, device="cuda:0")
                pos = 0
                for p, n in ddp_layout:
                    ddp_bucket[pos:pos + n].copy_(p.grad.reshape(-1))
                    pos += n
                dist.all_reduce(ddp_bucket, op=dist.ReduceOp.SUM)
                ddp_bucket /= world
                pos = 0
                for p, n in ddp_layout:
                    p.grad.copy_(ddp_bucket[pos:pos + n].view_as(p.grad))
                    pos += n
        elif args.ddp:
            # Average grads across ranks BEFORE clipping so every rank clips
            # and steps identically (bit-exact weights thereafter).
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= world
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if args.ddp and step == 1:
            # One-shot sync check: identical inits + averaged grads + identical
            # optimizer math must give bit-identical params on every rank.
            local = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
            dist.broadcast(local, src=0)
            assert torch.equal(local, torch.cat([p.detach().reshape(-1)
                                                 for p in model.parameters()])), "DDP rank divergence"

        if is_main and (step == 1 or step % 50 == 0):
            norms_a = (sum(a.detach().norm() for a in a_phys) / len(keys)).item()
            norms_b = (sum(b.detach().norm() for b in b_phys) / len(keys)).item()
            print(f"step {step:6d} domain={domain.split('__')[0]:<22s} loss={loss:.5f} "
                  f"|A|={norms_a:.2f} |B|={norms_b:.2f} lr={lr_at(step):.2e} "
                  f"({(time.time()-t0)/step:.2f}s/it)", flush=True)
        if is_main and (step % val_every == 0 or step == total_steps):
            model.eval()
            vls = {}
            with torch.inference_mode():
                for d in train_domains:
                    ev = val_evidence[d]
                    n_val = ev.shape[0]
                    # Level-domain val pools reach hundreds of episodes (vs ~13
                    # for lagfix) and each episode's predicted LoRA set is ~46M
                    # floats (~184MB fp32): materializing all per-episode outputs
                    # (let alone one full-batch forward) OOMs beside the JAX
                    # allocation. Accumulate a running SUM per chunk instead;
                    # sum/n equals the .mean(0) the consumer used.
                    sum_a = {k: None for k in keys}
                    sum_b = {k: None for k in keys}
                    for s in range(0, n_val, 32):
                        e = ev[s:s + 32]
                        w = val_views[d].expand(ev.shape[0], -1)[s:s + 32]
                        if args.variant == "concat":
                            cond = torch.cat(
                                [e, w.unsqueeze(1).expand(-1, e.shape[1], -1)], dim=-1)
                            p = model(cond)
                        else:
                            p = model(e, w)
                        for k, ab in p.items():
                            sa, sb = ab["A"].sum(0), ab["B"].sum(0)
                            sum_a[k] = sa if sum_a[k] is None else sum_a[k] + sa
                            sum_b[k] = sb if sum_b[k] is None else sum_b[k] + sb
                        del p
                    ab = lora_values([(sum_a[k] / n_val) * scale_a[i] for i, k in enumerate(keys)],
                                     [(sum_b[k] / n_val) * scale_b[i] for i, k in enumerate(keys)])
                    tls = []
                    for _ in range(2):  # 2 val batches per domain
                        vo, va = next(val_iters[d])
                        tls.append(float(backend.val_loss(
                            {p: jnp.asarray(t) for p, t in assemble(backend.mapping, ab).items()},
                            jax.random.PRNGKey(1), vo, va)))
                    # short name unless it collides (lighting_L3__deep would
                    # otherwise overwrite lighting_L3's entry in the same dict)
                    short = d.split("__")[0]
                    vls[d if short in vls else short] = sum(tls) / len(tls)
            mean_v = sum(vls.values()) / len(vls)
            history.append({"step": step, "val": vls, "val_mean": mean_v})
            print(f"[val] step {step} mean={mean_v:.5f} " +
                  " ".join(f"{k}={v:.4f}" for k, v in vls.items()), flush=True)

        if is_main and (step % save_every == 0 or step == total_steps):
            ckpt = args.output / f"step{step}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "representation": "direct_ab",
                "source": f"e2e_{args.variant}",
                "evidence": "dino_film_shared" if args.variant in ("film", "film_dropout")
                            else "dino_view_v4",
                "view_dropout": view_dropout,
                "state_dict": model.state_dict(),
                "model_args": model_args,
                "pose_feature": pose_cfg,
                "scales": scales, "modules": modules, "history": history,
                "oracle_paths": {d: str(p) for d, p in oracle_paths.items()},
                "domains": train_domains, "pooling": "patch", "variant": args.variant,
            }, ckpt)
            print(f"saved {ckpt}", flush=True)

    if is_main:
        print(f"done in {(time.time()-t0)/3600:.2f}h "
              f"({(time.time()-t0)/max(1, loop_steps):.3f}s/it); history tail: "
              f"{history[-1] if history else None}")
        if loss_log is not None:
            loss_log.close()
    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
