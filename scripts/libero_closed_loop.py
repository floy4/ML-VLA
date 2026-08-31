#!/usr/bin/env python3
"""LIBERO closed-loop eval of served LoRA policy variants (websocket client).

Pairs with scripts/serve_lora_policy.py.  Each policy variant is selected
per-request via the ``variant`` key; domains:

  clean -- unperturbed agentview camera
  view  -- agentview camera rotated +30 deg in azimuth about its lookat point
           (verified against the v1_azimuth30 demo frames)

Run with the policy server already up (libero env):

  MUJOCO_GL=egl python scripts/libero_closed_loop.py --host 127.0.0.1 --port 8123
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "open_the_middle_drawer_of_the_cabinet"
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = 300  # libero_goal, matches the OpenPI example
NUM_STEPS_WAIT = 10  # let objects settle
REPLAN_STEPS = 5
RESIZE = 224

# verified by scripts/72_verify_camera_perturbation.py (mae 6.38 vs demo frame)
VIEW_ANGLE_DEG = 30.0
VIEW_Z_PLANE = 1.08


def _quat2axisangle(quat):
    """Copied from the OpenPI LIBERO example (robosuite transform_utils)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def apply_view_perturbation(env) -> None:
    """Rotate the agentview camera +30 deg about its lookat point (in place).

    Must be re-applied after every env.reset(): robosuite hard_reset rebuilds
    the sim from XML and restores the default camera.
    """
    from robosuite.utils import transform_utils as tu

    sim = env.sim
    cam_id = sim.model.camera_name2id("agentview")
    pos = np.array(sim.model.cam_pos[cam_id], dtype=np.float64)
    quat = np.array(sim.model.cam_quat[cam_id], dtype=np.float64)  # w,x,y,z
    R = tu.quat2mat(np.array([quat[1], quat[2], quat[3], quat[0]]))
    fwd = R @ np.array([0.0, 0.0, -1.0])
    t = (VIEW_Z_PLANE - pos[2]) / fwd[2]
    target = pos + t * fwd

    theta = np.deg2rad(VIEW_ANGLE_DEG)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    new_pos = target + Rz @ (pos - target)

    fwd = target - new_pos
    fwd /= np.linalg.norm(fwd)
    z_cam = -fwd  # mujoco cameras look along local -z
    x_cam = np.cross(np.array([0.0, 0.0, 1.0]), z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    q = tu.mat2quat(np.stack([x_cam, y_cam, z_cam], axis=1))  # x,y,z,w
    sim.model.cam_pos[cam_id] = new_pos
    sim.model.cam_quat[cam_id] = np.array([q[3], q[0], q[1], q[2]])
    sim.forward()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_NAME, help="substring of the libero_goal task name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--episodes", type=int, default=10, help="episodes per (domain, variant)")
    parser.add_argument("--variants", nargs="+", default=["base"],
                        help="variant names as served by serve_lora_policy.py")
    parser.add_argument("--domains", nargs="+", default=["clean", "view"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video", action="store_true", help="save replay videos (first episode per group)")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "eval" / "m7_libero_closed_loop.json")
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _wcp

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task_id, task = next((i, t) for i, t in enumerate(suite.tasks) if args.task in t.name)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    initial_states = np.asarray(suite.get_task_init_states(task_id))
    task_description = str(task.language)
    print(f"task: {task_description} | {len(initial_states)} init states")

    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    client = _wcp.WebsocketClientPolicy(args.host, args.port)

    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        payload = {
            "task": task_description,
            "seed": args.seed,
            "episodes_per_group": args.episodes,
            "view_perturbation": {"angle_deg": VIEW_ANGLE_DEG, "z_plane": VIEW_Z_PLANE},
            "results": results,
            "summary": summarize(results),
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for domain in args.domains:
        for variant in args.variants:
            for episode in range(args.episodes):
                env.reset()
                if domain == "view":
                    apply_view_perturbation(env)
                obs = env.set_init_state(initial_states[episode])

                action_plan = collections.deque()
                frames = []
                t = 0
                done = False
                start = time.monotonic()
                while t < MAX_STEPS + NUM_STEPS_WAIT:
                    if t < NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, RESIZE, RESIZE))
                    if args.video and episode == 0:
                        frames.append(img)

                    if not action_plan:
                        element = {
                            "variant": variant,
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": task_description,
                        }
                        chunk = client.infer(element)["actions"]
                        action_plan.extend(np.asarray(chunk)[:REPLAN_STEPS].tolist())

                    obs, _, done, _ = env.step(action_plan.popleft())
                    if done:
                        break
                    t += 1

                final_success = bool(env.check_success())
                row = {
                    "domain": domain,
                    "variant": variant,
                    "episode": episode,
                    "init_state_index": episode,
                    "success": bool(done) or final_success,
                    "steps": t,
                    "duration_s": round(time.monotonic() - start, 1),
                }
                results.append(row)
                flush()
                print(
                    f"[{domain}/{variant}] ep{episode}: success={row['success']} "
                    f"steps={t} ({row['duration_s']}s)"
                )
                if args.video and episode == 0 and frames:
                    import imageio

                    path = args.output.parent / f"m7_video_{domain}_{variant}.mp4"
                    imageio.mimwrite(path, [np.asarray(f) for f in frames], fps=10)

    print(json.dumps(summarize(results), indent=2))
    flush()


def summarize(results: list[dict]) -> dict:
    summary = {}
    for domain in sorted({r["domain"] for r in results}):
        for variant in sorted({r["variant"] for r in results}):
            rows = [r for r in results if r["domain"] == domain and r["variant"] == variant]
            if not rows:
                continue
            summary[f"{domain}/{variant}"] = {
                "success_rate": sum(r["success"] for r in rows) / len(rows),
                "n": len(rows),
            }
    return summary


if __name__ == "__main__":
    main()
