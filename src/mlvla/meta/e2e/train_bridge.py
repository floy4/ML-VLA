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
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml

from mlvla.meta.e2e.lora_mapping import assemble, disassemble

VIEW_DIM = 7


def model_args_for(variant: str, module_shapes, hyper: dict) -> dict:
    """Constructor kwargs such that ``Cls(**model_args)`` reconstructs exactly.

    film/film_dropout add ``view_dim`` (absent from the yaml hypernet section);
    concat derives ``condition_dim = dino_dim + 7`` (V4 takes condition_dim,
    not dino_dim). Keys outside the target constructor's signature are dropped
    so no extra kwarg ever leaks into the checkpoint.
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
            "condition_dim": int(hyper["dino_dim"]) + VIEW_DIM,
            "module_shapes": module_shapes,
            "condition_hidden_dim": int(hyper["condition_hidden_dim"]),
            "hidden_dim": int(hyper["hidden_dim"]),
            "id_dim": int(hyper.get("id_dim", 256)),
            "trunk_layers": int(hyper.get("trunk_layers", 6)),
            "trunk_hidden": int(hyper.get("trunk_hidden", 1024)),
        }
    raise ValueError(f"unknown variant {variant!r}")


def build_model(variant: str, module_shapes, hyper: dict) -> torch.nn.Module:
    """film/film_dropout -> SharedFiLM; concat -> V4 direct head.

    Constructs from :func:`model_args_for` — the same parameterization that is
    saved into checkpoints, so saving and building can never drift apart.
    """
    from mlvla.meta.hypernet import SharedFiLMABHyperNetwork, V4DirectABHyperNetwork
    args = model_args_for(variant, module_shapes, hyper)
    if variant in ("film", "film_dropout"):
        return SharedFiLMABHyperNetwork(**args)
    if variant == "concat":
        return V4DirectABHyperNetwork(**args)
    raise ValueError(f"unknown variant {variant!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("film", "film_dropout", "concat"), required=True)
    parser.add_argument("--output", type=Path, required=True)   # /data2/.../e2e_<variant>/
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile() for meta-net")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=("default", "reduce-overhead", "max-autotune"),
                        help="torch.compile() mode (default: reduce-overhead)")
    parser.add_argument("--ddp", action="store_true",
                        help="Multi-process data parallel (torchrun): one GPU per "
                             "process, meta-net grads averaged via NCCL all-reduce. "
                             "Effective batch = batch_size * world_size.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    rung = cfg["rung"]                      # e.g. "fullvw4"
    train_domains = cfg["train_domains"]    # 8 domains (with __de2a6ce7 suffix)
    task = cfg["task_prompt"]               # "open the middle drawer of the cabinet"
    tr = cfg["training"]
    total_steps = args.steps or (500 if args.smoke else int(tr["max_steps"]))
    batch_size, lr, wd = int(tr["batch_size"]), float(tr["lr"]), float(tr["weight_decay"])
    warmup, clip, val_every, save_every = (int(tr["warmup"]), float(tr["clip"]),
                                           int(tr["val_every"]), int(tr["save_every"]))
    view_dropout = 0.3 if args.variant == "film_dropout" else 0.0
    evidence_k = int(tr.get("evidence_k", 8))

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
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        print(f"[ddp] world={world} rank={rank} effective_batch={batch_size * world}", flush=True)
    else:
        dist, rank, world = None, 0, 1
    is_main = rank == 0

    # ---- evidence ----
    from mlvla.meta.e2e.evidence import EvidenceBank
    feature_dir = Path(cfg["feature_cache_dir"])
    bank = EvidenceBank(feature_dir, train_domains)

    # ---- torch meta-net ----
    torch.manual_seed(0)
    model_args = model_args_for(args.variant, shapes, cfg["hypernet"])
    model = build_model(args.variant, shapes, cfg["hypernet"]).to("cuda:0")
    if args.compile:
        print(f"Compiling meta-net with torch.compile(mode={args.compile_mode!r})...", flush=True)
        model = torch.compile(model, mode=args.compile_mode)
        print("Compilation done.", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_at(step: int) -> float:
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
        # are NOT excluded from the train pool. Per-rank seed: DDP ranks must
        # not consume identical batches.
        loader_seed = 42 + 1000 * rank
        train_iters[domain] = backend.domain_loader(root, task, episode_ids=None,
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

    history = []
    rng_gen = torch.Generator().manual_seed(1000 * rank)
    t0 = time.time()
    for step in range(1, total_steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        domain = train_domains[int(torch.randint(len(train_domains), (1,), generator=rng_gen))]
        obs, act = next(train_iters[domain])

        x = bank.sample(domain, k=evidence_k, generator=rng_gen).to("cuda:0")
        if view_dropout > 0.0:
            keep = (torch.rand(x.shape[0], 1, 1, device="cuda:0") >= view_dropout).float()
            x = x * keep
        v = bank.view(domain).to("cuda:0").unsqueeze(0).expand(x.shape[0], -1)

        model.train()
        if args.variant == "concat":
            # view vector broadcast onto each of the 4 frames (same semantics
            # as view_params.append_view_params) -> [k,4,dino+7]; V4 frame-means.
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
        module_ab = lora_values(a_phys, b_phys)
        tensors = {p: jnp.asarray(t) for p, t in assemble(backend.mapping, module_ab).items()}
        key = jax.random.fold_in(jax.random.PRNGKey(0), step)
        loss, grads = backend.loss_and_grad(dict(tensors), key, obs, act)
        loss = float(loss)
        if not math.isfinite(loss):
            raise RuntimeError(f"step {step}: non-finite loss")
        gmod = disassemble(backend.mapping, {p: np.asarray(g) for p, g in grads.items()})
        ga, gb = lora_grads(gmod)
        if not all(torch.isfinite(g).all() for g in ga + gb):
            raise RuntimeError(f"step {step}: non-finite lora grads")

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
        if args.ddp:
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
                    n_val = val_evidence[d].shape[0]
                    vv = val_views[d].expand(n_val, -1)
                    if args.variant == "concat":
                        cond = torch.cat(
                            [val_evidence[d], vv.unsqueeze(1).expand(-1, val_evidence[d].shape[1], -1)],
                            dim=-1)
                        preds = model(cond)
                    else:
                        preds = model(val_evidence[d], vv)
                    ab = lora_values([preds[k]["A"].mean(0) * scale_a[i] for i, k in enumerate(keys)],
                                     [preds[k]["B"].mean(0) * scale_b[i] for i, k in enumerate(keys)])
                    tls = []
                    for _ in range(2):  # 2 val batches per domain
                        vo, va = next(val_iters[d])
                        tls.append(float(backend.val_loss(
                            {p: jnp.asarray(t) for p, t in assemble(backend.mapping, ab).items()},
                            jax.random.PRNGKey(1), vo, va)))
                    vls[d.split("__")[0]] = sum(tls) / len(tls)
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
                "scales": scales, "modules": modules, "history": history,
                "oracle_paths": {d: str(p) for d, p in oracle_paths.items()},
                "domains": train_domains, "pooling": "patch", "variant": args.variant,
            }, ckpt)
            print(f"saved {ckpt}", flush=True)

    if is_main:
        print(f"done in {(time.time()-t0)/3600:.2f}h; history tail: "
              f"{history[-1] if history else None}")
    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
