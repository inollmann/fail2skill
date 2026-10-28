# panda_cosmos_al

Data pipeline for an active-learning loop around NVIDIA Cosmos 3 with a simulated **Franka Emika Panda**
in **Isaac Lab 3.0 / Isaac Sim 6.2**:

* **Manual episodes**: drive the simulated Panda with **MoveIt 2** (ROS 2 Humble). It uses the
  `multipanda_ros2` description and MoveIt config, with the same controller interfaces as the real robot.
* **Automatic episodes**: a scripted expert reads the object pose from the simulator and plans a grasp.
  Optional **corruptions** (grasp offset, early release, joint jitter, ...) make it fail on purpose.
* **Dashcam**: every episode, successful or not, is saved with three camera videos, the executed
  actions, robot/object state and the exact initial state.
* **Replay**: re-execute any action sequence (e.g. one proposed by Cosmos) from a failed episode's initial
  state. The simulator decides whether the "fixed" episode really succeeds.

Everything runs in Docker. **Continuing this work? Start with [docs/HANDOVER.md](docs/HANDOVER.md).** See [cosmos/README.md](cosmos/README.md) for how the data maps onto Cosmos 3.

```text
 ┌──────────── sim container (Isaac Lab, Python 3.12) ────────────┐     ┌──── ros container (Humble) ─────┐
 │ PandaAL-PickPlace-v0 ── dashcam ──> data/episodes/...           │     │ move_group + RViz               │
 │   ▲ 8-D joint-position actions                                  │ TCP │ al_isaac_bridge:                │
 │   ├─ scripts/teleop_moveit.py  (bridge server) <────────────────┼─────┤  /joint_states, /clock          │
 │   ├─ scripts/generate_episodes.py (scripted expert + noise)     │JSON │  panda_arm_controller/follow_…  │
 │   └─ scripts/replay_episode.py (recorded or Cosmos actions)     │     │  panda_gripper/gripper_action   │
 └─────────────────────────────────────────────────────────────────┘     └─────────────────────────────────┘
```

## Requirements

* Linux, a recent NVIDIA driver (developed with 580; Blackwell cards such as the RTX 5070 Ti need ≥ 570),
  Docker with the NVIDIA Container Toolkit.
* An **RTX** GPU for the simulator. Isaac Sim renders the cameras with RTX, which needs RT cores.
  **H100s have none**, so use the A6000s (or the 5070 Ti) for simulation and the H100 server for Cosmos.
* About 70 GB of disk for the images (the Isaac Sim base image alone is ~32 GB).

## Quick start

```bash
git clone <this repo> panda_cosmos_al && cd panda_cosmos_al
./al.sh setup          # clones Isaac Lab next to this repo (../IsaacLab) at the pinned commit
./al.sh build          # isaac-lab-base + panda-al-sim + panda-al-ros (first build takes a while)
./al.sh test           # unit tests (no simulator)
./al.sh generate --num_episodes 4 --run_name first_try    # first run also compiles shaders: be patient
```

### Manual episodes with MoveIt

```bash
./al.sh teleop         # add --gui to also open the Isaac Sim window
```

This starts the simulator and MoveIt/RViz in the background and opens the **episode console**:

| key | action |
| --- | --- |
| `o` / `c` | open / close the gripper |
| `s` / `f` / `d` | end the episode as success / failure / discard it |
| `q` | quit (stops the containers) |

In RViz, drag the interactive marker and *Plan & Execute* with planning group `panda_arm`. Use `o`/`c`
for the gripper, or plan the `hand` group. The cubes are shown as markers and the table is a collision
object. An episode also ends by itself when the red cube rests on the blue one with the gripper open
(success), or after 120 s (failure). `./al.sh logs` shows the simulator and MoveIt output.

The bridge also speaks plain ROS, so scripts can drive it like the real robot:
`move_group`'s `/move_action`, `/panda_arm_controller/follow_joint_trajectory`,
`/panda_gripper/gripper_action`, and the `/al/episode/*` and `/al/gripper/*` `std_srvs/Trigger` services.

### Automatic episodes

```bash
./al.sh generate --num_envs 8 --num_episodes 200 --corrupt_ratio 0.8 --run_name batch01
./al.sh generate --num_episodes 50 --corrupt_ratio 0 --run_name clean01            # clean demos
./al.sh generate --num_episodes 50 --modes miss_grasp drop_in_transit --run_name f01
```

The failure modes are defined in `CORRUPTION_MODES` in `src/panda_al/scripted.py`: `miss_grasp`,
`high_grasp`, `yaw_misalign`, `drop_in_transit`, `misplace`, `jitter`, `rushed`. The sampled parameters
are stored in each episode's `meta.json`. A corrupted episode that still succeeds is filed as a success,
because the label always comes from the simulator, not from the injected corruption.

### Multi-task environment (objects, tasks, prompts)

`PandaAL-MultiTask-v0` draws a task, its objects, their poses and a language prompt per env at every reset, from two
registries that are plain files (see [docs/automated_collection.md](docs/automated_collection.md)):

* `assets/objects/<name>/object.yaml`: one folder per object (USD, noun and color for prompts, size, optional mass and
  friction). **Adding an asset = adding a folder.**
* `configs/tasks/<name>.yaml`: goal (`stack`, `beside` left/right), filters for the pick and place objects, prompt
  templates such as `"put the {pick.color} {pick.noun} on top of the {place.color} {place.noun}"`, time limit, weight.

```bash
./al.sh registry                                        # validate both and list what they define
./al.sh generate --task PandaAL-MultiTask-v0 --num_envs 8 --num_episodes 100 --run_name multitask
./al.sh replay --episodes data/episodes/multitask/failure/*   # picks the right env and task by itself
```

Each episode's `meta.json` has the prompt as `instruction` and the task details under `extra.task` (task, prompt,
pick/place object, goal, objects on the table). The original `PandaAL-PickPlace-v0` is unchanged.
`scripts/benchmark_sim.py` and `scripts/benchmark_policy.py` measure simulator throughput and policy latency.

### Unattended failure collection (VLA in the loop)

`./al.sh collect <campaign.yaml>` lets an array of robots work through the multi-task env, controlled by a served
policy, and keeps **only the failures** (for Cosmos 3). Everything is set in a campaign file
([configs/campaigns/example.yaml](configs/campaigns/example.yaml)): policy server and adapter, task weights, cameras,
obstacle probability, what to save, quotas and limits. Plan and design: [docs/automated_collection.md](docs/automated_collection.md).

```bash
./al.sh serve lerobot --checkpoint <dir|repo> &      # or openpi's serve_policy.py, here or on the Jetson Thor
./al.sh collect configs/campaigns/example.yaml       # runs for hours; Ctrl-C to stop, run again to continue
./al.sh collect configs/campaigns/dryrun_expert.yaml # pipeline test without any VLA (scripted expert as the "policy")
```

* Every finished episode is judged by the simulator; successes are dropped **without being encoded** (frames are
  buffered in RAM and only failures are written, by background threads).
* Failures get automatic tags: `stalled_at_start`, `no_grasp`, `dropped`, `misplaced`, `collision`, `knocked_off`,
  `timeout` (in `meta.json` → `extra.failure_tags`). Failures with only `drop_tags` (default `stalled_at_start`)
  are not saved, and per-task / per-tag quotas keep one failure type from dominating.
* The run directory `$AL_DATA/episodes/<name>/` contains the episodes plus `manifest.jsonl` (saved episodes: task,
  prompt, tags, checkpoint label, path) and `stats.jsonl` (all episodes including successes, for per-task success
  rates, which are the weakness signal for the next fine-tuning round).
* Robust by design: the same command resumes from `stats.jsonl`/`manifest.jsonl`; a crashed simulator is restarted
  (`AL_COLLECT_MAX_RESTARTS`, default 20); a lost policy server is re-tried for about a minute, then the run stops
  and the supervisor tries again after 30 s.
* Rendering is the cost driver: every camera in the scene costs about 40 ms per control step and 2 GB of GPU memory at
  16 envs. List only the cameras you need in `cameras:`; all others (except the ones the policy adapter reads) are
  removed from the scene.

### Fine-tuning a policy on demonstrations

```bash
./al.sh collect configs/campaigns/demos.yaml                                      # 450 scripted demonstrations (13 min)
./al.sh export-lerobot --source data/episodes/demos --out data/lerobot/demos_v0   # LeRobotDataset (slow: ~25 min)
./al.sh train --dataset data/lerobot/demos_v0 --name smolvla_iter0 --steps 8000    # prints the serve command at the end
./al.sh serve lerobot --checkpoint /workspace/data/checkpoints/smolvla_iter0/checkpoints/last/pretrained_model ...
./al.sh collect configs/campaigns/eval_smolvla.yaml && ./al.sh report data/episodes/eval_smolvla_iter0
```

`--source <run dir>:3` in `export-lerobot` counts that source three times (extra weight for the failure cases that Cosmos
turned into verified successes). Checkpoints are written every 2000 steps; serve and evaluate earlier ones the same way
to get a ladder of competence. First result (8000 steps): 2 % success, the gripper reaches the cubes but misses them by a
median 8 cm, see [docs/automated_collection.md](docs/automated_collection.md).

### Obstacle failures

The corruption mode `obstacle_cylinder` stands a static cylinder (height 0.18 m, radius = cube side / 2) on the
table between the two cubes. It is placed right after the env reset has randomized the cube poses, at a fraction
0.4 - 0.6 along the line from the pick to the place cube and up to 5 cm to the side. The expert plan is not
changed, so the episode only fails if the robot (or the carried cube) really touches the cylinder. Any contact
ends the episode with the `obstacle_collision` termination and it is filed under `failure/`. The cylinder is
stored in `meta.json` → `extra.corruption` (`obstacle_radius`, `obstacle_height`, `obstacle_along`,
`obstacle_lateral`, resulting `obstacle_position` in the robot base frame) and in `initial_state.npz`, so replays
contain it too. In all other episodes it waits under the table, out of sight.

```bash
./al.sh generate --modes obstacle_cylinder --corrupt_ratio 1 --num_episodes 20 --run_name obstacle
```

### Replay / validate actions

```bash
./al.sh replay --episodes data/episodes/batch01/failure/*                       # reproduce
./al.sh replay --episodes data/episodes/batch01/failure/<id> \
               --actions data/cosmos_actions.npy --source synthetic --run_name cosmos_iter1
```

`--actions` must be a `(T, 8)` `.npy` in the action format below and must live under `data/` so the
container can see it.

### Policy rollouts

Any policy that controls the robot can be dropped into the loop. It runs as a **policy server** in its own
environment (container, machine, framework), and the simulator talks to it over websocket using
[openpi](https://github.com/Physical-Intelligence/openpi)'s protocol: an observation dict goes in, an action chunk
comes back. Every rollout is recorded by the dashcam like any other episode (`source: policy`, server and adapter
settings in `meta.json` → `extra.policy`), so policy failures go straight to Cosmos.

```bash
# pi0.5 (zero-shot DROID policy: Franka + ZED mini wrist camera, like this setup), in an openpi checkout:
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid          # serves on port 8000
./al.sh policy --adapter droid --num_episodes 20 --run_name pi05_droid

# LeRobot policies (SmolVLA, ACT, diffusion, ...), e.g. a SmolVLA fine-tuned on our episodes:
./al.sh build policy
./al.sh serve lerobot --checkpoint /workspace/data/checkpoints/smolvla_iter0
./al.sh policy --adapter native --num_episodes 20 --run_name smolvla_iter0

./al.sh serve hold     # smoke test: a policy that holds still (needs no checkpoint)
```

The policy server can run on another machine (e.g. the H100 server; policies need no RT cores): set
`AL_POLICY_HOST` / `AL_POLICY_PORT` in `docker/.env.local` or pass `--host` / `--port`.

**Batching**: our servers (`./al.sh serve lerobot|openpi|hold`) also answer *batched* requests, one model call for many
envs (`PolicyClient.infer_batch`; the collector does this automatically). SmolVLA: 99 ms for one env, 367 ms for 16.
A policy must keep no state between observations for this (chunk-based policies don't). For a server on another
machine set `jpeg_quality: 90` (campaign) so images travel as JPEG (30x smaller). openpi's own `serve_policy.py`
answers single requests only; to batch pi0.5, run `python -m panda_al.policy.serve openpi --config pi05_droid
--checkpoint gs://openpi-assets/checkpoints/pi05_droid` from an openpi checkout with this repo's `src/` on the
`PYTHONPATH` (this wrapper has not been tested against real pi0.5 weights yet).

The **adapter** translates between the simulator and a policy family (`src/panda_al/policy/adapters.py`):

| adapter | request | response `actions` |
| --- | --- | --- |
| `native` | `observation/images/<cam>` (640x360 RGB), `observation/joint_position` (7), `observation/gripper_position` (1), `observation/state` (8), `prompt` | `(H, 8)` in the env's action format |
| `droid` | openpi's DROID keys: `exterior_image_1_left`, `wrist_image_left` (224x224, padded), joint + gripper position, `prompt` | `(H, 8)` joint velocities in [-1, 1] + gripper; target = `q + 0.2 v` per step as in DROID |

The first `--open_loop_horizon` actions of each chunk are executed (default 8, as in openpi's DROID example),
then the policy is queried again. A new policy family needs a new adapter class, and for frameworks other than
openpi or LeRobot also a wrapper with `infer(obs) -> {"actions": ...}` served by `panda_al.policy.serve_policy`.

## Custom assets

`assets/` holds the table (`panda_table.usd`) and the ZED mini hand mount (`Franka_Hand_ZED_mini_Mount.usdc`).
`scripts/build_assets.py` derives from the mount CAD the sim-ready `zed_mini.usd` (body, two lenses, left/right USD
cameras with ZED mini intrinsics) and `panda_hand_zed_mount.usda` (mount + camera in metres). The `wrist` /
`wrist_right` cameras are the ZED mini's left/right lens, `wrist_depth` is its depth (valid 0.1 - 15 m). The mount's
pose on the hand is `MOUNT_POS` / `MOUNT_ROT` in `src/panda_al/tasks/pick_place_env_cfg.py`.
Regenerate after changing the CAD: `docker compose ... run --rm -v ./assets:/workspace/panda_cosmos_al/assets sim python scripts/build_assets.py`
(`assets/` is mounted read-only in the containers by default).

## Episode format

```text
data/episodes/<run_name>/{success,failure}/<episode_id>/
    meta.json            task, instruction, outcome, termination terms, source, corruption, ...
    data.npz             action (T,8), joint_position (T,7), gripper_position (T,1), cartesian_position (T,6),
                         ee_quat (T,4), object_pose (T,2,7), joint_velocity, timestamp, is_first, is_last
    initial_state.npz    exact simulator state at t=0 (for replay)
    videos/exterior_1.mp4 exterior_2.mp4 wrist.mp4 wrist_right.mp4     640x360, 15 fps, one frame per row
    videos/wrist_depth.mkv                             lossless 16-bit depth of the wrist camera [mm], 0 = invalid
```

* **Action** = 7 absolute arm joint targets [rad] + gripper (0 open, 1 closed), i.e. what MoveIt or the IK
  expert commanded. This is the same 8-D `joint_pos` action space as the Cosmos 3 DROID recipe.
* Row `t` is the observation before `action[t]`. The last row is the terminal observation.
* Episodes can be read without Isaac Lab: `from panda_al.episode_io import load_episode` (needs `numpy`, `av`).

## Running on the shared machines

Put per-user settings in `docker/.env.local` (not committed):

```bash
AL_GPU=1                   # which GPU (nvidia-smi index) this user's containers get
AL_BRIDGE_PORT=45220       # containers use host networking: one port per user
ROS_DOMAIN_ID=17           # and one ROS domain per user
AL_DATA=/data/<you>/al     # optional: where episodes go
```

To use both A6000s for generation, run one generator per GPU with different run names and seeds:

```bash
AL_GPU=0 ./al.sh generate --num_envs 8 --num_episodes 500 --run_name a_gpu0 --seed 0 &
AL_GPU=1 ./al.sh generate --num_envs 8 --num_episodes 500 --run_name a_gpu1 --seed 1 &
```

GUIs (RViz, `--gui`) need a local X display. Over SSH, use X forwarding or a remote desktop session.

## Where things live

| path | what |
| --- | --- |
| `src/panda_al/tasks/pick_place_env_cfg.py` | the task: scene, cameras, render quality, 15 Hz control, actions, success check |
| `src/panda_al/dashcam.py` | recorder term that writes episodes (runs inside `env.step()`) |
| `src/panda_al/episode_io.py` | episode format, writer, loader (no Isaac dependencies) |
| `src/panda_al/scripted.py` | scripted expert + failure modes |
| `src/panda_al/sim_bridge.py` | simulator side of the MoveIt bridge (TCP/JSON protocol) |
| `src/panda_al/policy/` | policy deployment: websocket client/server (openpi protocol), adapters, LeRobot wrapper |
| `ros/al_ros_bridge/` | ROS 2 package: bridge node, episode console, MoveIt launch file |
| `scripts/` | `teleop_moveit.py`, `generate_episodes.py`, `replay_episode.py`, `run_policy.py`, `build_assets.py` |
| `docker/` | Dockerfiles, compose file, `.env` |

`src/`, `scripts/` and `tests/` are bind-mounted into the sim container, so code changes need no
rebuild. Changes to `ros/` need `./al.sh build ros`.

To build the ROS image from a local multipanda checkout instead of upstream GitHub, set
`MULTIPANDA_CONTEXT=/abs/path/to/multipanda_ros2` in `docker/.env.local`.

## Design notes

* **Why not ROS inside Isaac Lab?** Isaac Sim 6 runs Python 3.12 and ROS 2 Humble uses 3.10. The simulator
  speaks a tiny JSON protocol and the ROS container owns everything ROS, so the MoveIt side can use exactly
  the same packages as the real robot.
* **Why trajectory-level instead of a 1 kHz ros2_control loop?** The Isaac Lab env steps at 15 Hz (DROID's
  rate) with cameras. MoveIt hands over whole time-parameterised trajectories, which the simulator plays
  back against simulation time, so the sim can run slower than real time without breaking execution.
* The robot is Isaac Lab's Franka asset with stiff PD gains and no gravity sag (as in Isaac Lab's IK tasks).
  The gripper is binary in simulation.
* The scene reuses Isaac Lab's maintained cube-stacking scene (`isaaclab_tasks.contrib.stack`), so upgrades
  to Isaac Lab carry over. To change the task, start from `pick_place_env_cfg.py`.
