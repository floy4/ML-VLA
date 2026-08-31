"""Replay LIBERO-goal demos under perturbed conditions, writing LeRobot v3.0 parquet shards.

Each invocation handles one condition end-to-end (50 demos x 10 tasks).
For parallel runs, the outer launcher sets CUDA_VISIBLE_DEVICES per process.

Run in the `libero` conda env (MUJOCO_GL=egl is set below).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import pathlib
import re
import shutil
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image
import torch

from mlvla import paths as _paths

_paths.add_to_sys_path()

PERTURBED_ROOT = pathlib.Path(os.environ.get(
    "MLVLA_PERTURBED_OUT_ROOT",
    os.environ.get("WIZARD_PERTURBED_OUT_ROOT", _paths.get("perturbed_root")),
))
SOURCE_ROOT = pathlib.Path(_paths.get("source_dataset"))
_LIBERO_PLUS = pathlib.Path(_paths.get("libero_plus_root"))
BDDL_DIR = _LIBERO_PLUS / "libero" / "libero" / "bddl_files" / "libero_goal"
INIT_DIR = _LIBERO_PLUS / "libero" / "libero" / "init_files" / "libero_goal"

TASKS = [
    ("open_the_middle_drawer_of_the_cabinet",   "open the middle drawer of the cabinet"),
    ("open_the_top_drawer_and_put_the_bowl_inside", "open the top drawer and put the bowl inside"),
    ("push_the_plate_to_the_front_of_the_stove", "push the plate to the front of the stove"),
    ("put_the_bowl_on_the_plate",               "put the bowl on the plate"),
    ("put_the_bowl_on_the_stove",               "put the bowl on the stove"),
    ("put_the_bowl_on_top_of_the_cabinet",      "put the bowl on top of the cabinet"),
    ("put_the_cream_cheese_in_the_bowl",        "put the cream cheese in the bowl"),
    ("put_the_wine_bottle_on_the_rack",         "put the wine bottle on the rack"),
    ("put_the_wine_bottle_on_top_of_the_cabinet", "put the wine bottle on top of the cabinet"),
    ("turn_on_the_stove",                       "turn on the stove"),
]

FPS = 10
CHUNK_SIZE = 1000

_condition_id = {"id": "unknown"}


def _quat2axisangle(quat):
    """Quaternion (w-last) -> axis-angle, same convention as the source data."""
    quat = list(quat)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return np.asarray(quat[:3]) * 2.0 * math.acos(quat[3]) / den


def load_source_actions(task_desc):
    meta_table = pq.read_table(SOURCE_ROOT / "meta" / "tasks.parquet")
    task_to_idx = {r["task"]: r["task_index"] for r in meta_table.to_pylist()}
    if task_desc not in task_to_idx:
        raise ValueError("Task not in source tasks.parquet: " + repr(task_desc))

    ep_table = pq.read_table(
        SOURCE_ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        columns=["episode_index", "tasks", "data/chunk_index", "data/file_index"],
    )
    rows = ep_table.to_pylist()
    target_eps = [r["episode_index"] for r in rows if r["tasks"] and r["tasks"][0] == task_desc]
    if not target_eps:
        raise ValueError("No episodes for task " + repr(task_desc))
    target_set = set(target_eps)

    chunk_dir = SOURCE_ROOT / "data" / "chunk-000"
    shard_files = sorted(chunk_dir.glob("file-*.parquet"))
    actions_by_ep = {}
    for shard in shard_files:
        t = pq.read_table(shard, columns=["episode_index", "actions", "frame_index"])
        eps = t.column("episode_index").to_pylist()
        acts = t.column("actions").to_pylist()
        frms = t.column("frame_index").to_pylist()
        for ep, a, fr in zip(eps, acts, frms):
            if ep in target_set:
                actions_by_ep.setdefault(ep, []).append((fr, a))
    fixed = {}
    for ep, pairs in actions_by_ep.items():
        pairs.sort(key=lambda x: x[0])  # sort by frame_index
        arr = np.asarray([a for _, a in pairs], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 7)
        fixed[ep] = arr
    sorted_eps = sorted(fixed.keys())
    return [fixed[ep] for ep in sorted_eps], sorted_eps


def load_init_states(task_stem):
    p = INIT_DIR / (task_stem + ".pruned_init")
    states = torch.load(str(p), map_location="cpu")
    return np.asarray(states)


def register_light_problem_class(cond_id, scene_xml_path):
    from libero.libero.envs.problems.libero_tabletop_manipulation import (
        Libero_Tabletop_Manipulation,
        register_problem,
    )
    from libero.libero.envs.regions import REGION_SAMPLERS, TableRegionSampler
    old_libero = Libero_Tabletop_Manipulation is None
    if old_libero:
        # Old-LIBERO checkout (what `libero` resolves to inside eval
        # processes): its @register_problem returns None, so the module
        # attribute is None. The real class lives in TASK_MAPPING.
        from libero.libero.envs.bddl_base_domain import TASK_MAPPING
        Libero_Tabletop_Manipulation = TASK_MAPPING["libero_tabletop_manipulation"]
    class_name = "Libero_Tabletop_Manipulation_wizard_perturbed_" + cond_id

    def make_init(scene_xml, old_libero):
        def _init(self, bddl_file_name, horizon_view=None, vertical_view=None,
                  init_state=None, *args, **kwargs):
            if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
                kwargs["scene_xml"] = str(scene_xml)
            if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
                kwargs["scene_properties"] = {
                    "floor_style": "light-gray",
                    "wall_style": "light-gray-plaster",
                }
            # LIBERO-plus's wrapper passes horizon/vertical/init_state
            # positionally and its base consumes them; old LIBERO's wrapper
            # never sends them and its base would choke on the kwargs.
            fwd = {} if old_libero else {
                "horizon_view": horizon_view,
                "vertical_view": vertical_view,
                "init_state": init_state,
            }
            Libero_Tabletop_Manipulation.__init__(
                self,
                bddl_file_name=bddl_file_name,
                *args, **fwd, **kwargs,
            )
        return _init

    # Build the subclass with the base's own metaclass: in the closed-loop
    # eval process (both LIBERO and LIBERO-plus on sys.path) the resolved
    # base class may carry ABCMeta, and plain type() then raises
    # "metaclass conflict".
    cls = type(Libero_Tabletop_Manipulation)(
        class_name,
        (Libero_Tabletop_Manipulation,),
        {"__init__": make_init(str(scene_xml_path), old_libero)},
    )
    register_problem(cls)
    REGION_SAMPLERS[class_name.lower()] = {"table": TableRegionSampler}
    return class_name


def write_bddl_with_problem_class(base_bddl_path, problem_class_name, out_path):
    text = base_bddl_path.read_text()
    new_text = re.sub(
        r"\(define\s+\(problem\s+[^\)]+\)",
        "(define (problem " + problem_class_name + ")",
        text,
        count=1,
    )
    out_path.write_text(new_text)


def get_view_bddl_path(base_bddl_stem, view):
    suffix = view["bddl_filename_suffix"]
    bddl_path = BDDL_DIR / (base_bddl_stem + suffix + ".bddl")
    if not bddl_path.exists():
        base_bddl = BDDL_DIR / (base_bddl_stem + ".bddl")
        bddl_path.symlink_to(base_bddl)
    return bddl_path


def build_env_bddl(condition, base_bddl_stem):
    if condition["type"] in ("view", "combined"):
        bddl_path = get_view_bddl_path(base_bddl_stem, condition["view"])
    else:
        bddl_path = BDDL_DIR / (base_bddl_stem + ".bddl")

    if condition["type"] in ("light", "combined"):
        cls_name = register_light_problem_class(
            _condition_id["id"], condition["light"]["scene_xml"]
        )
        # For combined conditions, the derived filename MUST include the
        # `_view_<h>_<v>_<s>_<er>_<ev>_initstate_<i>` suffix — env_wrapper
        # parses the filename to extract camera rotation parameters. Without
        # it, the light-registered class gets horizon_view=0, vertical_view=0
        # and the view perturbation is silently dropped.
        if condition["type"] == "combined":
            view_suffix = condition["view"]["bddl_filename_suffix"]
        else:
            view_suffix = ""
        derived = BDDL_DIR / (
            "_wizard_perturbed_" + _condition_id["id"] + "_" +
            base_bddl_stem + view_suffix + ".bddl"
        )
        write_bddl_with_problem_class(bddl_path, cls_name, derived)
        if view_suffix:
            # LIBERO-plus's wrapper splits "_view_..." off the filename and
            # then reads BDDL content from the STRIPPED path — both files
            # must exist and carry the perturbed problem class.
            stripped = BDDL_DIR / (
                "_wizard_perturbed_" + _condition_id["id"] + "_" +
                base_bddl_stem + ".bddl"
            )
            write_bddl_with_problem_class(bddl_path, cls_name, stripped)
        bddl_path = derived

    return bddl_path


def _png_bytes(arr_uint8_hwc):
    buf = io.BytesIO()
    Image.fromarray(arr_uint8_hwc).save(buf, format="png")
    return buf.getvalue()


def episode_to_table(frames):
    image_struct_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    image_rows = [{"bytes": _png_bytes(f["agentview_image"]), "path": None} for f in frames]
    wrist_rows = [{"bytes": _png_bytes(f["wrist_image"]), "path": None} for f in frames]
    return pa.table({
        "frame_index": pa.array([f["frame_index"] for f in frames], type=pa.int32()),
        "episode_index": pa.array([f["episode_index"] for f in frames], type=pa.int32()),
        "task_index": pa.array([f["task_index"] for f in frames], type=pa.int32()),
        "index": pa.array([f["global_index"] for f in frames], type=pa.int64()),
        "timestamp": pa.array([f["timestamp"] for f in frames], type=pa.float32()),
        "image": pa.array(image_rows, type=image_struct_type),
        "wrist_image": pa.array(wrist_rows, type=image_struct_type),
        "state": pa.array([f["state"].tolist() for f in frames], type=pa.list_(pa.float32(), 8)),
        "actions": pa.array([f["actions"].tolist() for f in frames], type=pa.list_(pa.float32(), 7)),
    })


def write_lerobot_meta(cond_root, episodes_meta, total_frames, task_descriptions):
    meta_dir = cond_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    pq.write_table(
        pa.table({
            "task_index": pa.array(list(range(len(task_descriptions))), type=pa.int32()),
            "task": pa.array(task_descriptions, type=pa.string()),
        }),
        meta_dir / "tasks.parquet",
    )

    ep_table = pa.table({
        "episode_index": pa.array([e["episode_index"] for e in episodes_meta], type=pa.int32()),
        "tasks": pa.array([[e["task"]] for e in episodes_meta], type=pa.list_(pa.string())),
        "length": pa.array([e["length"] for e in episodes_meta], type=pa.int32()),
        "task_index": pa.array([e["task_index"] for e in episodes_meta], type=pa.int32()),
    })
    pq.write_table(ep_table, meta_dir / "episodes" / "chunk-000" / "file-000.parquet")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": len(task_descriptions),
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": "0:" + str(len(episodes_meta))},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "image":         {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channel"], "fps": FPS},
            "wrist_image":   {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channel"], "fps": FPS},
            "state":         {"dtype": "float32", "shape": [8], "names": ["state"], "fps": FPS},
            "actions":       {"dtype": "float32", "shape": [7], "names": ["actions"], "fps": FPS},
            "episode_index": {"dtype": "int32", "shape": [1], "names": None, "fps": FPS},
            "frame_index":   {"dtype": "int32", "shape": [1], "names": None, "fps": FPS},
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None, "fps": FPS},
            "task_index":    {"dtype": "int32", "shape": [1], "names": None, "fps": FPS},
            "index":         {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))


def _step_env(env, action):
    """Wrap env.step to handle both gym API styles: (obs, reward, done, info) or obs."""
    result = env.step(action)
    if isinstance(result, tuple):
        return result[0]
    return result


def run_smoke(cid, task_stem, all_conds):
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    condition = all_conds[cid]
    task_desc = dict(TASKS)[task_stem]
    _condition_id["id"] = cid + "_smoke"
    base_bddl = BDDL_DIR / (task_stem + ".bddl")
    init_states = load_init_states(task_stem)

    env0 = OffScreenRenderEnv(bddl_file_name=str(base_bddl), camera_heights=256, camera_widths=256)
    env0.reset(); env0.set_init_state(init_states[0])
    obs0 = _step_env(env0, np.zeros(7)); env0.close()

    perturbed_bddl = build_env_bddl(condition, task_stem)
    env1 = OffScreenRenderEnv(bddl_file_name=str(perturbed_bddl), camera_heights=256, camera_widths=256)
    env1.reset(); env1.set_init_state(init_states[0])
    obs1 = _step_env(env1, np.zeros(7)); env1.close()

    out_dir = PERTURBED_ROOT / "_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (256 * 2 + 30, 256), (0, 0, 0))
    canvas.paste(Image.fromarray(obs0["agentview_image"]), (0, 0))
    canvas.paste(Image.fromarray(obs1["agentview_image"]), (256 + 30, 0))
    out = out_dir / (cid + "_" + task_stem + ".png")
    canvas.save(out)
    diff = float(np.abs(obs0["agentview_image"].astype(float) - obs1["agentview_image"].astype(float)).mean())
    print("Saved " + str(out) + " (mean pixel diff: " + str(round(diff, 2)) + ")")


def run_condition(cid, condition, task_stems, num_demos):
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    _condition_id["id"] = cid
    cond_root = PERTURBED_ROOT / cid
    cond_root.mkdir(parents=True, exist_ok=True)
    # Archive the modified lighting scene next to the data (verification checks it).
    if condition.get("light") and condition["light"].get("scene_xml"):
        _scene_src = pathlib.Path(condition["light"]["scene_xml"])
        if _scene_src.exists():
            shutil.copy2(_scene_src, cond_root / "scene.xml")
    data_dir = cond_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    episodes_meta = []
    task_descs_seen = []
    global_idx = 0
    ep_global = 0

    for task_idx, (task_stem, task_desc) in enumerate(TASKS):
        if task_stem not in task_stems:
            continue
        task_descs_seen.append(task_desc)
        print("[" + cid + "] Loading source actions for " + task_stem + "...", flush=True)
        actions_per_demo, src_eps = load_source_actions(task_desc)
        n = min(num_demos, len(actions_per_demo))
        if n < num_demos:
            print("[" + cid + "]   WARN: only " + str(len(actions_per_demo)) + " demos available (asked " + str(num_demos) + ")")
        print("[" + cid + "]   got " + str(len(actions_per_demo)) + " demos; using " + str(n), flush=True)
        init_states = load_init_states(task_stem)
        if len(init_states) < n:
            raise RuntimeError(task_stem + " has " + str(len(init_states)) + " init_states, need " + str(n))

        bddl_path = build_env_bddl(condition, task_stem)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl_path),
                                 camera_heights=256, camera_widths=256)
        try:
            for demo_i in range(n):
                actions = actions_per_demo[demo_i]
                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise RuntimeError("bad action shape " + str(actions.shape) + " for demo " + str(demo_i))
                # Deterministic replay first. Physics drift vs the original
                # collection env makes ~39% of source demos fail under our
                # build; retry those with a gain sweep (systematic
                # under/overshoot correction) then escalating action noise,
                # several seeds per scale. We store the EXECUTED actions so
                # (obs, action) stays self-consistent even for modified
                # attempts. Gripper dim untouched.
                attempts = ([("exact", 0.0)]
                            + [("gain", g) for g in (1.05, 0.95, 1.10, 0.90)]
                            + [("noise", s) for s in (0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60)
                               for _ in range(3)])
                action_std = np.array([0.3931, 0.3505, 0.4914, 0.0550, 0.0777, 0.1041, 0.0],
                                      dtype=np.float32)
                for attempt, (kind, param) in enumerate(attempts):
                    env.reset()
                    # set_init_state returns the observation OF the init state
                    # (env_wrapper.py regenerate_obs_from_state -> _get_observations).
                    obs = env.set_init_state(init_states[demo_i])
                    exec_actions = actions
                    if kind == "gain":
                        exec_actions = actions.copy()
                        exec_actions[:, :6] *= param
                        exec_actions = np.clip(exec_actions, -1.0, 1.0).astype(np.float32)
                    elif kind == "noise":
                        rng = np.random.RandomState(
                            (0xC0FFEE ^ (demo_i * 2654435761 % (2**31)) ^ (attempt * 40503))
                            % (2**31))
                        exec_actions = np.clip(
                            actions + rng.randn(*actions.shape).astype(np.float32)
                            * action_std * param, -1.0, 1.0).astype(np.float32)
                    frames = []
                    for t in range(len(exec_actions)):
                        # Record the observation BEFORE executing the action, matching
                        # the original LIBERO hdf5 convention (official collect script
                        # deletes the trailing after-action state precisely to align
                        # (s_t, a_t); see libero_100_collect_demonstrations.py:173).
                        # Recording obs AFTER env.step pairs (s_{t+1}, a_t) — a
                        # constant one-step lag vs the source data.
                        state = np.concatenate([
                            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                            _quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
                            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                        ])
                        # MuJoCo EGL offscreen returns images rotated 180° vs the
                        # top-left convention of the source dataset.
                        # Apply rot180 (both axes flip) to match.
                        av_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8)
                        wr_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8)
                        frames.append({
                            "agentview_image": av_img,
                            "wrist_image": wr_img,
                            "state": state,
                            "actions": exec_actions[t].astype(np.float32),
                            "frame_index": t,
                            "episode_index": ep_global,
                            "task_index": task_idx,
                            "timestamp": t / FPS,
                        })
                        obs = _step_env(env, exec_actions[t])
                    ok = bool(env.check_success())
                    if ok or attempt == len(attempts) - 1:
                        break
                if attempt > 0 or not ok:
                    print("[" + cid + "]   " + task_stem + ": demo " + str(demo_i)
                          + " attempt=" + str(attempt) + " (" + kind + " " + str(param) + ")"
                          + " success=" + str(ok), flush=True)
                for fr in frames:
                    fr["global_index"] = global_idx
                    global_idx += 1
                all_frames.extend(frames)
                episodes_meta.append({
                    "episode_index": ep_global,
                    "task": task_desc,
                    "length": len(frames),
                    "task_index": task_idx,
                    "success": ok,
                    "attempt": attempt,
                    "attempt_kind": kind,
                    "attempt_param": param,
                })
                ep_global += 1
                if demo_i % 10 == 0:
                    print("[" + cid + "]   " + task_stem + ": demo " + str(demo_i) + "/" + str(n) + " (" + str(len(frames)) + " frames)", flush=True)
        finally:
            env.close()

        if all_frames:
            backup = data_dir / ("_partial_" + task_stem + ".parquet")
            pq.write_table(episode_to_table(all_frames), backup)

    table = episode_to_table(all_frames)
    pq.write_table(table, data_dir / "file-000.parquet")
    for p in data_dir.glob("_partial_*.parquet"):
        p.unlink()
    write_lerobot_meta(cond_root, episodes_meta, len(all_frames), task_descs_seen)
    (cond_root / "replay_success.json").write_text(json.dumps(
        [{"episode_index": e["episode_index"], "task": e["task"],
          "success": e["success"], "length": e["length"],
          "attempt": e.get("attempt", 0),
          "attempt_kind": e.get("attempt_kind", "exact"),
          "attempt_param": e.get("attempt_param", 0.0)} for e in episodes_meta],
        indent=1))
    print("[" + cid + "] Wrote " + str(cond_root) + ": " + str(len(episodes_meta)) + " episodes, " + str(len(all_frames)) + " frames")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", default="", help="Comma-separated condition IDs; empty = all 9")
    p.add_argument("--tasks", default="", help="Comma-separated task stems; empty = all 10")
    p.add_argument("--smoke", action="store_true", help="Run 1 demo x 1 task, save PNG, exit")
    p.add_argument("--num_demos", type=int, default=50)
    args = p.parse_args(argv)

    conds_yaml = yaml.safe_load((PERTURBED_ROOT / "conditions.yaml").read_text())
    all_conds = conds_yaml["conditions"]
    selected_conds = args.conditions.split(",") if args.conditions else list(all_conds)
    selected_tasks = args.tasks.split(",") if args.tasks else [t[0] for t in TASKS]

    if args.smoke:
        assert len(selected_conds) == 1 and len(selected_tasks) == 1, \
            "--smoke requires exactly one condition and one task"
        run_smoke(selected_conds[0], selected_tasks[0], all_conds)
        return

    for cid in selected_conds:
        run_condition(cid, all_conds[cid], selected_tasks, args.num_demos)


if __name__ == "__main__":
    main()
