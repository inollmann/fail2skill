# Automated failure collection with a VLA

Status: M0-M4 implemented (M0 measured on the development machine only; the A6000s and the Thor are still to measure), first SmolVLA baseline trained but not competent (2 % success), M5 and M6 not started. **Hand-over and next steps: [HANDOVER.md](HANDOVER.md).** This is an additional data source next to teleop, the scripted
generator, replay and single-policy rollouts (`./al.sh policy`). All of those stay as they are.

## Goal

A long-running collection mode in which an array of simulated Pandas works through a rotating set of tasks, driven
by a VLA (pi0.5 or SmolVLA) behind one shared policy server. Every episode is judged by the simulator and **only
failures are saved**. They go to Cosmos 3 (analyst and teacher), the sim replays Cosmos' corrected actions, and
verified successes are mixed into the next fine-tuning set with extra weight on the policy's weaknesses.

```text
 collect -> failures -> Cosmos 3 (analysis + corrected actions) -> sim replay (verifies) -> synthetic successes
    ^                                                                                              |
    +---- fine-tuned policy <---- dataset mix, weighted by per-task failure rate <-----------------+
```

## Architecture

```text
 Isaac Sim process (GPU0)                    Policy server (GPU1, or Jetson Thor over LAN)
 PandaAL-MultiTask-v0, N envs in a grid      pi0.5 (openpi) / SmolVLA (LeRobot)
  env i: task_i, prompt_i, objects_i         batched inference: (B obs) -> (B, H, 8)
  collect.py -- one batched request per step -->
             <-- action chunks for the envs whose chunk ran out --
  dashcam -> failures only -> data/episodes/<campaign>/ + manifest.jsonl, stats.jsonl
```

* One sim process per GPU, N envs per process on a grid (Isaac Lab cloning).
* One policy server for all envs. Each control step the collector sends one request containing the envs whose
  action chunk is used up. The sim waits for the policy (inference latency does not change behaviour, episodes stay
  reproducible). Optional latency emulation can come later as a failure source.
* The policy server speaks the existing openpi-style websocket protocol (`src/panda_al/policy`), extended with a
  batch dimension in M3.

### Where to host the VLA

| setup | sim | VLA | notes |
| --- | --- | --- | --- |
| A (start here) | GPU0 | GPU1 | localhost, no network. pi0.5 needs about 8 GB, SmolVLA about 2 GB. |
| B | GPU0 and GPU1, one sim each | Jetson Thor | about twice the sim throughput; inference latency close to a real deployment on a Thor. |

Risks of B: openpi is JAX first and JAX on Thor (aarch64, Blackwell) is unproven; its PyTorch path and LeRobot
(PyTorch) should run in NVIDIA's Jetson PyTorch containers. M0 checks this before anything depends on it.

**Network (Gigabit LAN):** the sim side resizes images to the policy's input size, so a DROID request is about
0.3 MB (2 x 224x224x3). Full 640x360 frames (about 0.7 MB each) are never sent raw. Real Gigabit throughput is
about 100 MB/s, so bandwidth is not the limit; model latency on the Thor is (measured in M0).

## What gets saved

* **Failures only.** Successes are discarded when the episode ends (`keep_success_fraction` defaults to 0).
* Every finished episode, successful or not, adds one line to `stats.jsonl` (task, prompt, objects, checkpoint,
  outcome, steps, no video). The per-task / per-checkpoint success rate is the weakness signal of the adaptive loop
  and cannot be recovered from failures alone.
* Failures are tagged automatically from the recorded state: `stalled_at_start`, `no_grasp`, `dropped`,
  `misplaced`, `collision`, `knocked_off`, `timeout`. Uninformative ones (the robot never moves) can be dropped or
  downsampled, and quotas apply per task and per tag so one failure type does not dominate.
* `initial_state.npz` is stored as before, so the sim can replay any failure from its exact start.

## Tasks, prompts and assets (M1)

Everything that varies is data, not code.

```text
assets/objects/<name>/object.yaml   one folder per object: USD, noun and color for prompts, scale, mass, friction
configs/tasks/<name>.yaml           one file per task: goal type, roles, prompt templates, time limit
```

* A **task** names a goal (`stack`, `beside`, ...), the roles (`pick`, `place`) with filters on object properties,
  and prompt templates using nouns: `"put the {pick.color} {pick.noun} on the {place.color} {place.noun}"`.
  A phrasing is drawn per episode and stored in the episode metadata.
* **Adding an asset**: drop a folder with the USD and an `object.yaml` into `assets/objects/`; the next run picks it
  up (`./al.sh assets check` validates scale, collision, mass, resting stability). The three cubes are the first
  entries. Adding a task that uses existing goal types is one YAML file; a new success condition needs one small
  Python function in the goal registry.
* Objects enter the scene through a pool: every registry object exists in each env and is parked out of sight under
  the table when it is not part of the current episode (works up to roughly 10 object types; beyond that Isaac Lab's
  multi-asset spawner assigns assets per env at spawn).
* The environment's reset event draws a task per env, the objects and their roles, non-overlapping poses and a
  prompt. Success checks dispatch on the per-env task, so different envs can run different tasks at the same time.

## Milestones

* **M0: measure.** `scripts/benchmark_sim.py` (sim throughput vs. number of envs, with or without recording) and
  `scripts/benchmark_policy.py` (round-trip latency of a policy server). Remaining: run both on the A6000s, run the
  policy benchmark against pi0.5 and SmolVLA on the A6000 and on the Thor, and check that the VLA stacks run on
  the Thor. (Isaac Lab's `TiledCameraCfg` is deprecated: `Camera` already uses tiled rendering, so there is
  nothing to compare.)
* **M1: multi-task environment (implemented).** `PandaAL-MultiTask-v0` with object registry (`assets/objects`),
  task registry (`configs/tasks`: `stack`, `beside_left`, `beside_right`), goal functions, per-env prompts stored in
  `meta.json` (`extra.task`, and `instruction`), scripted expert generalised to the task set, replay restoring the
  task, `./al.sh registry` to validate definitions. Not yet done: a simulator-based asset check (resting stability,
  collision shapes), non-box objects for the scripted expert (it grasps top-down using the object's yaw).
* **M2: collector (implemented).** `./al.sh collect <campaign.yaml>` (`scripts/collect.py`, `src/panda_al/collect/`):
  * campaign files (`configs/campaigns/`), validated at start-up: driver (`policy` or `expert` for dry runs and
    baselines), task weights, cameras, obstacle perturbation, save rules, quotas, limits;
  * **failure-only saving without wasted work**: the dashcam buffers frames in RAM (`defer_encoding`) and a thread
    pool encodes only the episodes the collector keeps; `episode_hook` lets the collector tag, filter and count each
    finished episode before anything is written;
  * automatic failure tags (`collect/tags.py`), per-task / per-tag quotas, `drop_tags`, `manifest.jsonl` and
    `stats.jsonl` (`collect/ledger.py`), resume from those files;
  * robustness: a supervisor in `al.sh` restarts the simulator after a crash (status file, since the Isaac launcher
    hides exit codes), `PolicyRunner` reconnects to a lost policy server and otherwise the run stops with status 75
    and is retried, numerically invalid episodes are discarded;
  * not yet: lighting / friction perturbations (only the obstacle exists), a 24 h soak test, batched inference (M3).
* **M3: batched policy server (implemented).** `PolicyClient.infer_batch` sends the observations of many envs in one
  request (`{"batched": true, "observations": [...]}`, answered with `actions (B, H, D)`); the same websocket server
  still answers plain openpi requests. Servers announce `batched` in their metadata; the collector uses one connection
  and one request per control step for exactly the envs whose chunk ran out (`BatchedPolicyRunner`), and falls back to
  one connection per env for servers without it.
  * `serve lerobot`: one model call per batch (policies with a single observation step: SmolVLA, pi0/pi0.5 ports, ACT).
  * `serve openpi`: `OpenPiPolicy` wraps openpi's `Policy` (input transforms per observation, one stacked model call
    padded to a bucket size to avoid JAX recompiles, output transforms per result). It relies on private attributes of
    openpi's `Policy` and has only been tested against a stand-in, not against real pi0.5 weights.
  * Several clients (e.g. one simulator per GPU) may share one server; model calls are serialised by a lock.
  * `jpeg_quality` (adapter / campaign): images travel as JPEG, 30x smaller (22 KB instead of 675 KB per 640x360
    frame, 1 ms to encode), which a remote server over Gigabit needs: raw, a request for 16 envs and 3 cameras is
    about 32 MB, i.e. 0.3 s on the wire.
* **M4: near-competent policies (pipeline implemented, first baseline trained; not yet competent).**
  * demonstrations: `configs/campaigns/demos.yaml` (scripted expert, clean, successes only, 150 per task, positions,
    colors, distractors and phrasings randomised): 450 episodes in 13 minutes at 16 envs;
  * `./al.sh export-lerobot` (`panda_al/export/lerobot_export.py`): episodes -> LeRobotDataset v3 (state = 7 joints +
    gripper closure, action = the env's 8-D action, terminal row dropped, `--source dir:N` repeats a source N times
    for emphasis, `export_info.json` lists every exported episode). Slow: 26 minutes for 450 episodes (the video
    re-encoding of LeRobot dominates; fine for now);
  * `./al.sh train` (`panda_al/train.py`): `lerobot-train` for SmolVLA with our cameras renamed to `camera1..3`,
    checkpoints every `--save_freq` steps (the ladder), prints the matching `./al.sh serve` command;
  * `./al.sh report <run dirs>` (`scripts/report_stats.py`): success rate per task with Wilson 95 % intervals, failure
    tags, per checkpoint label (`--by_label`); `configs/campaigns/eval_smolvla.yaml` evaluates a served checkpoint;
  * the policy image now has LeRobot's training extra and ffmpeg (10 GB); `./al.sh test-policy` runs the tests that
    need it.
* **M5: adaptive loop.** Collector task weights follow the measured success rate and aim for a learning band
  (about 20-80 %); tasks above the band are sampled less, tasks near 0 % get a smaller share. Iteration ledger:
  checkpoint -> per-task success rate -> dataset mix. Dataset builder weights failures by task and tag.
* **M6: Thor deployment.** JetPack packaging of the policy server, benchmark, setup B.

## M0 results (development machine: RTX 5070 Ti 16 GB, 28 CPU cores, 61 GB RAM)

Scripted expert driving `PandaAL-MultiTask-v0` with the four cameras of the dashcam (`benchmark_sim.py`):

| envs | no recording | recording (dashcam on) |
| --- | --- | --- |
| 1 | 44 ms / control step | |
| 4 | 48 ms | 164 ms (24 env-steps/s) |
| 8 | | 221 ms (36 env-steps/s) |
| 16 | 47 ms | 315 ms (51 env-steps/s, 20 episodes/min); 308 ms without depth |
| 24 | out of memory | |

What this says:

* **Recording is what makes the sim slow**, because it reads all cameras every step. Without it nothing reads the
  cameras and the left column is only physics, so it does not predict collection speed.
* **Cost per camera in the scene**: 16 envs with deferred encoding take 253 / 206 / 170 / 130 ms per control step with
  4 / 3 / 2 / 1 cameras, i.e. about 40 ms (2.5 ms per env) and 2.1 GB of GPU memory per camera. Cameras that are
  merely not *recorded* still cost the same, they have to be removed from the scene (`restrict_cameras`; the
  campaign's `cameras:` list does that). The fixed cost is about 90 ms.
* **Deferred encoding** (frames buffered in RAM, only failures encoded in background threads, successes never
  encoded) takes 16 envs from 308 to 253 ms per step (about 18 %). The camera read-back, not the video codec,
  remains the main cost.
* **GPU memory**: about 4.4 GB base plus about 0.55 GB per env with four cameras (a 16 GB card fits 16 envs, 24 ran out
  of memory). With three cameras 16 envs need 11.3 GB, with two 9.2 GB. A 48 GB A6000 should fit roughly 70 envs;
  to be measured there.
* **Throughput** with the three cameras of the DROID-style setup: 78 env-steps/s (30 episodes/min at 16 envs for
  the 117-step scripted episodes). A VLA that only needs two cameras (exterior + wrist) reaches 94 env-steps/s.
* NVENC (`h264_nvenc`) is available in PyAV if video encoding becomes a bottleneck; so far it is not.

Policy side (`benchmark_policy.py`, loopback, "hold" policy): a request with three 640x360 images is 2.0 MB, a DROID
request 0.3 MB. Over Gigabit that is about 16 ms and 3 ms of transfer, so resizing on the sim side (the DROID
adapter does) matters. SmolVLA base: about 100 ms per chunk on this GPU once warm (batch 1, direct call). One
batched call per 8 control steps for 16 envs is roughly 20 ms per step amortised, against about 270 ms of simulation
per step at 16 envs, so rendering, not the VLA, limits throughput here; batching still saves up to about 2x versus
sequential calls.

## M3 results (SmolVLA base, same development machine, policy server and client on one host)

`benchmark_policy.py --adapter native --state_dim 6 --batch_sizes 1 2 4 8 16` (3 cameras at 640x360 per observation):

| envs per request | latency | per env | request size (raw) |
| --- | --- | --- | --- |
| 1 | 99 ms | 99 ms | 2.0 MB |
| 2 | 112 ms | 56 ms | 4.0 MB |
| 4 | 142 ms | 35 ms | 7.9 MB |
| 8 | 202 ms | 25 ms | 15.8 MB |
| 16 | 367 ms | 23 ms | 31.6 MB |

One call for 16 envs is 4.3x cheaper per env than 16 sequential calls, and the gain flattens beyond 8 envs on this GPU.
With JPEG the same requests carry far less data and the server's latency is unchanged (364 ms at 16 envs; the test
images are random noise, real frames compress better). In the collector, with every env asking for a chunk every 8th
step, about 2-3 envs ask per step and one request serves them. Rendering (about 200 ms per step at 16 envs) stays
the larger cost; the policy is a minor one once batched.

## M4 results: first baseline (SmolVLA base fine-tuned on 450 scripted demonstrations)

Training: 8000 steps, batch 16 (2.5 epochs of 51,261 frames), 3.35 steps/s on the 5070 Ti (about 40 minutes, 7 GB),
100M trainable parameters (the action expert; the VLM is frozen). The loss fell from 1.6 to 0.06 without a plateau.
Checkpoints at 2000 / 4000 / 6000 / 8000 steps.

Evaluation (`eval_smolvla.yaml`, 120 episodes on layouts the demonstrations did not use, 12 envs, 11 minutes):

| task | episodes | success | main failure |
| --- | --- | --- | --- |
| stack | 30 | 0 % | no_grasp, timeout |
| beside_left | 43 | 0 % | no_grasp, timeout |
| beside_right | 47 | 4 % | no_grasp, timeout |
| all | 120 | 2 % (95 % CI 0-6 %) | `no_grasp` in 115 of 118 failures |

The policy is not competent yet, but it is not doing nothing either: in every episode the hand goes down to the
table and the gripper closes in two thirds of those steps. It misses the cube: the median horizontal distance of the
gripper to the cube at its lowest point is 8 cm (25 % of the episodes: under 4 cm, 9 %: under 2 cm). So it has learned
the gross motion, but not yet to localise a 4.7 cm cube precisely from the cameras. Expected levers, none of them tried
yet: more training (the loss is still falling), more demonstrations, a larger batch, unfreezing more of the model.
The 118 failures of this run are the first genuinely policy-made failure set (`data/episodes/eval_smolvla_iter0`).
