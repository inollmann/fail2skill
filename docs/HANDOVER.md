# Hand-over: failure collection with a VLA, Cosmos 3 as analyst and teacher

You are taking over a working pipeline whose first trained policy is **not good enough yet**. This document says what
exists, how to run it on the machine with the two A6000s, what the numbers on the development machine were, what to do
next (in order), and which traps cost time. Read it together with [automated_collection.md](automated_collection.md)
(plan, design and measurements) and the [README](../README.md) (commands).

## 1. The goal in one picture

```text
  VLA (SmolVLA / pi0.5)  ──controls──>  many simulated Pandas (Isaac Lab, RTX GPU)   ← "collect"
        ▲                                        │ judged by the simulator, only FAILURES are saved
        │                                        ▼
  fine-tune with extra weight            failed episodes (video + actions + exact initial state)
  on the weak tasks                              │
        ▲                                        ▼
        │                              Cosmos 3: explains the failure, proposes corrected actions   ← NOT BUILT
  verified synthetic successes  <──  replay in the simulator from the failure's initial state
                                       (`scripts/replay_episode.py --actions ...`): only real successes count
```

The policy is meant to be *any* policy that can drive the robot. Cosmos 3 is the analyst and the data generator, not the
policy. Successful episodes are deliberately **not** saved by the collector (only counted in `stats.jsonl`), because the
dataset is for turning failures into successes.

## 2. Status

| part | state |
| --- | --- |
| Simulator task, 3 cameras + ZED mini stereo/depth on a custom mount, table, obstacle failure mode | done, used daily |
| Multi-task environment `PandaAL-MultiTask-v0` (object/task registries, prompts per env) | done |
| Collector `./al.sh collect` (failure-only saving, tags, quotas, resume, supervisor, batched policy calls) | done, soak-tested only for minutes (crash/resume and dead-server paths tested, no 24 h run) |
| Policy servers `./al.sh serve lerobot` (batched), `serve openpi` (batched wrapper) | LeRobot tested with SmolVLA; **the openpi wrapper was never run against real pi0.5 weights** |
| Demonstrations, LeRobot export, fine-tuning wrapper, evaluation tooling | done |
| **A competent policy** | **no**: first SmolVLA baseline succeeds in 2 % of episodes (section 5) |
| Cosmos 3 integration (analysis, generating corrected actions) | **not built**; only the data contract exists ([cosmos/README.md](../cosmos/README.md)) and replay validation works |
| Adaptive weighting by success rate, iteration ledger (M5) | **not built** (see 6.F). I stopped because it needs a model with success rates in the middle of the range |
| Benchmarks on the A6000s, anything on the Jetson Thor (M0 remainder, M6) | **not done** |

Everything is in the working tree of this repository; check `git log` / `git status` for what has been committed.
`data/` is git-ignored and **not** part of the repository (section 5).

## 3. Getting running on the A6000 machine

Prerequisites: Linux, NVIDIA driver (580 was used), Docker with the NVIDIA Container Toolkit, ~120 GB disk (images:
Isaac Sim base 32 GB, `panda-al-sim` 50 GB, `panda-al-policy` 10 GB, ROS 5 GB).
**Isaac Sim needs RT cores** (the A6000s have them; the H100 server does not: use it for Cosmos only).

```bash
git clone <repo> panda_cosmos_al && cd panda_cosmos_al
./al.sh setup                 # clones Isaac Lab at the pinned commit next to the repo
./al.sh build                 # isaac-lab-base + sim + ros images (long)
./al.sh build policy          # policy server / trainer image
cat > docker/.env.local <<'EOF'
AL_GPU=0                      # simulator on the first A6000
AL_POLICY_GPU=1               # VLA server / training on the second
AL_BRIDGE_PORT=45210          # host networking: one port per user on a shared machine
ROS_DOMAIN_ID=42
EOF
```

Check that it works, in this order (each step takes minutes; the first simulator start compiles shaders):

```bash
./al.sh test && ./al.sh test-policy                       # 45 + 15 unit tests, no GPU needed
./al.sh registry                                          # objects and tasks load
./al.sh collect configs/campaigns/dryrun_expert.yaml      # whole pipeline without a VLA; ends by itself, ~3 min
./al.sh run isaaclab -p scripts/benchmark_sim.py --num_envs 16 --record --defer --cameras exterior_1 exterior_2 wrist
```

The dry run must finish with "all task quotas reached" and leave `data/episodes/dryrun_expert/{failure,manifest.jsonl,
stats.jsonl}`. On the development machine (RTX 5070 Ti, 16 GB) the last command gave 206 ms per control step.

## 4. Everyday commands

| I want to | command |
| --- | --- |
| scripted demonstrations (successes only) | `./al.sh collect configs/campaigns/demos.yaml` (450 episodes, 13 min at 16 envs) |
| episodes -> LeRobot dataset | `./al.sh export-lerobot --source data/episodes/demos --out data/lerobot/demos_v0` (slow: 26 min) |
| fine-tune SmolVLA | `./al.sh train --dataset data/lerobot/demos_v0 --name smolvla_iter0 --steps 8000` |
| serve a checkpoint | the command `./al.sh train` prints at its end (`serve lerobot --checkpoint ... --rename ...`) |
| evaluate it / collect its failures | edit `configs/campaigns/eval_smolvla.yaml` (label!), `./al.sh collect <it>`, `./al.sh report data/episodes/<name>` |
| unattended failure collection | `configs/campaigns/example.yaml` (quotas, obstacle, limits), same command |
| replay a failure, or Cosmos' proposed actions | `./al.sh replay --episodes <failed dir> [--actions x.npy --source synthetic --run_name cosmos_iter1]` |
| teleoperate with MoveIt / old scripted generator | `./al.sh teleop --gui` / `./al.sh generate ...` (README) |
| add an object / task | folder in `assets/objects/<name>/object.yaml` / file in `configs/tasks/`; `./al.sh registry` |

A campaign file is the unit of work: it names the run directory (`data/episodes/<name>/`), and running the same file
again **resumes** from `stats.jsonl`/`manifest.jsonl`. Change `name` for a new run. The supervisor restarts the
simulator after crashes (`AL_COLLECT_MAX_RESTARTS`); it stops at once for an invalid campaign.

## 5. The baseline and what is (not) in `data/`

* 450 scripted demonstrations (seed 100) -> `data/lerobot/demos_v0` -> SmolVLA base fine-tuned 8000 steps, batch 16
  (2.5 epochs, 40 min, 7 GB, 100M trainable parameters: the VLM is frozen), loss 1.6 -> 0.06 and still falling.
* Evaluation, 120 episodes on unseen layouts (seed 7): **2 % success** (stack 0/30, beside_left 0/43, beside_right
  2/47). In 115 of 118 failures the tag is `no_grasp`. The hand reaches the table in *every* episode and closes the
  gripper, but its median horizontal distance to the cube is **8 cm** (9 % of episodes within 2 cm): it has not learned to
  localise the cube. So the failures are "reach but miss", not yet the near misses Cosmos should learn from.
* Not in git, regenerate or copy: `data/episodes/demos` (201 MB), `data/lerobot/demos_v0` (268 MB),
  `data/checkpoints/smolvla_iter0` (5 GB, checkpoints at 2000/4000/6000/8000 steps),
  `data/episodes/eval_smolvla_iter0` (144 MB, 118 failures). Regenerating costs about 80 minutes in total.
* Speeds on the development GPU (to be re-measured on the A6000s): simulator 206 ms per control step at 16 envs with
  3 cameras (every camera in the scene costs about 40 ms and 2 GB at 16 envs: keep only the ones you need, the
  campaign's `cameras:` does that); GPU memory about 0.55 GB per env; SmolVLA 99 ms for one env, 367 ms for a batch
  of 16; training 3.35 steps/s at batch 16.

## 6. What to do next, in order

**A. Re-baseline on the A6000s (half a day).** Run the checks of section 3, then `benchmark_sim.py` for 8/16/32/64
envs (`--record --defer --cameras exterior_1 exterior_2 wrist`) to find the env count where throughput stops growing
(rendering saturates; 48 GB should fit about 70 envs), and `benchmark_policy.py` against SmolVLA on the second
A6000 with `--batch_sizes 1 4 16 32 --jpeg_quality 90`. Put the numbers into `automated_collection.md` ("M0 results").

**B. Get a competent policy. This blocks everything after it.** The goal is per-task success of roughly 20-80 % (a
policy that mostly works but still fails often), the "learning band" the plan assumes. Things to try, cheapest first,
and evaluate each checkpoint with 120+ episodes per run (`./al.sh report`, CIs are wide below 100):
1. train much longer: 30-50k steps and a larger batch (batch 16 used 7 GB on the 16 GB development card, so 32-64
   should fit the A6000s). The loss had not plateaued at 8000 steps;
2. more demonstrations (`demos.yaml`, `quota per_task: 500`); they are cheap, the export is the slow part;
3. look at the ladder: serve the 2000/4000/6000/8000 checkpoints and compare, to see whether more steps help;
4. if the grasp stays imprecise: unfreeze more of the model (`./al.sh train ... -- --policy.freeze_vision_encoder=false
   --policy.train_expert_only=false`; both are on by default), or change what the model sees (SmolVLA resizes images
   to 512x512 with padding, so our 640x360 frames become 512x288 and a 4.7 cm cube is a few tens of pixels);
5. the DROID-style zero-shot pi0.5 is also worth a baseline (D below).

**C. Cosmos 3 loop (the point of the project).** Contract: [cosmos/README.md](../cosmos/README.md). Per failed episode
Cosmos gets the videos, the prompt, the tags and the recorded actions, and must return a `(T, 8)` action array. Validate
it with `./al.sh replay --episodes <failed dir> --actions out.npy --source synthetic --run_name cosmos_iterN`: the
simulator replays it from the failure's exact initial state and files the result under `success/` or `failure/` with
`parent_episode` set. Only those verified successes go into fine-tuning. This script already works; the part that
produces `out.npy` does not exist. Keep the verified fraction per task and tag: it is the real signal of how useful
Cosmos is.

**D. pi0.5.** (1) zero-shot with openpi's own server and the `droid` adapter (`./al.sh policy --adapter droid`), which
has the same observation keys and DROID action semantics as the real robot setup; (2) batched via
`python -m panda_al.policy.serve openpi --config pi05_droid --checkpoint ...` run from an openpi checkout with this
repo's `src/` on `PYTHONPATH` (the policy package should also run on openpi's Python 3.11; untested). `OpenPiPolicy` in
`src/panda_al/policy/serve.py` relies on private attributes of openpi's `Policy` and has **only been tested against a
stand-in** (`tests/test_policy.py`); expect to fix it, and compare its batched output with single `infer` calls.

**E. Jetson Thor (M6).** The Thor only runs a policy server (Isaac Sim stays on the A6000s). `docker/Dockerfile.policy`
is x86 (`python:3.12-slim` + pip torch); build an aarch64 variant on NVIDIA's Jetson PyTorch base image, install
`lerobot[smolvla]==0.6.1`, and run `python -m panda_al.policy.serve` with `PYTHONPATH=src` (the `panda_al.policy` package
needs only numpy, msgpack, websockets, pillow). Whether JAX/openpi runs on it is unknown; LeRobot (PyTorch) should.
Over the Gigabit link use `jpeg_quality: 90` (a raw 16-env request is about 32 MB; with JPEG about 1 MB) and run
`benchmark_policy.py` from the simulator host against it before anything else.

**F. M5, adaptive weighting (only after B).** What is missing, concretely:
1. `ResetTaskScene` (`src/panda_al/tasks/multitask.py`) draws tasks with `self.weights`; add an
   `env.set_task_weights(dict)` hook (I mentioned it in the plan, it does **not** exist) and call it from
   `scripts/collect.py` every N finished episodes;
2. weights from `Ledger.success_rates()` over a sliding window (`collect/ledger.py` has the counts): full weight inside
   the 20-80 % band, reduced above (already solved) and below (hopeless, uninformative failures), with a floor;
3. an iteration ledger (checkpoint label -> per-task success -> dataset mix) as a JSON file next to the run;
4. a dataset builder that turns per-task / per-tag failure rates into the `--source dir:N` repeat counts of
   `export-lerobot` (that flag exists).
Do not tune the band on the 2 % baseline: with every task at 0-4 % the weighting cannot be evaluated.

**G. Smaller things.** Perturbations other than the obstacle (lighting, friction, camera jitter) are in the plan but
not implemented; the scripted expert only handles box-like objects (top-down grasp from the object's yaw), so non-box
assets need an expert of their own or teleoperation; a 24 h soak test of `collect` has not been run;
`CYLINDER_OBSTACLE_NOTE.md` in the repo root is a saved chat summary and can be deleted.

## 7. Where things are

| path | what |
| --- | --- |
| `al.sh`, `docker/` | the entry point; compose services `sim`, `policy`, `moveit`, `teleop-sim` |
| `src/panda_al/tasks/` | `pick_place_env_cfg.py` (original single task: scene, cameras, table, ZED mount, obstacle, success), `multitask_env_cfg.py` + `multitask.py` (registries, per-env tasks, success terms) |
| `src/panda_al/{objects,taskspec,goals}.py` | object / task registries and goal functions (pure Python + torch) |
| `src/panda_al/dashcam.py`, `episode_io.py` | recorder (deferred encoding, episode hook), episode format and writer |
| `src/panda_al/scripted.py`, `obstacle.py` | scripted expert and its failure modes, obstacle placement |
| `src/panda_al/policy/` | websocket client/server (openpi protocol + batching), adapters (`native`, `droid`), runners, `serve.py` |
| `src/panda_al/collect/` | `campaign.py`, `ledger.py`, `tags.py`, `drivers.py`; the loop is `scripts/collect.py` |
| `src/panda_al/export/lerobot_export.py`, `train.py` | dataset export, fine-tuning wrapper |
| `scripts/` | `collect.py`, `run_policy.py`, `generate_episodes.py`, `replay_episode.py`, `benchmark_*.py`, `report_stats.py`, `build_assets.py`, `check_registry.py` |
| `assets/`, `configs/` | custom USD assets (table, ZED mini mount and camera), object and task registries, campaigns |
| `tests/` | unit tests (`./al.sh test`, `./al.sh test-policy`) |

## 8. Traps (each of these cost time)

* **The Isaac launcher hides exit codes**: `isaaclab -p` and `python.sh` turn any failure into exit 1 and closing the
  Kit app exits 0. `collect.py` therefore writes its status to `$AL_COLLECT_STATUS_FILE` for the supervisor in `al.sh`;
  no file means "crashed, restart".
* **Env config modules are imported before the Kit app starts.** Importing `isaaclab.envs.ManagerBasedRLEnv` or
  anything that loads `pxr` there breaks Kit start-up with hundreds of "undefined symbol ...pxrInternal" errors. Use
  `TYPE_CHECKING` imports (see `multitask.py`).
* **Manager terms**: Isaac Lab inspects the signature of a term's `__call__`; list every param explicitly with a default
  (the manager fills unset ones with `None`, treat that as "not given").
* **Cameras are rendered whenever any is read**, and cost the same whether recorded or not: remove unused cameras from the
  scene (`restrict_cameras`), do not only drop them from the recorder.
* **Do not edit `al.sh` while an `./al.sh` command is running**: bash reads scripts incrementally and executes garbage.
* **The policy container runs as root.** `train`/`export` set `umask 0` so you can delete their output; older
  root-owned files: `docker run --rm --entrypoint rm -v $PWD/data:/d panda-al-policy -rf /d/<path>`.
* PyTorch DataLoader workers need shared memory: the policy service has `shm_size: 8gb`; with the default 64 MB
  training dies with "unable to allocate shared memory".
* **Batching needs a stateless policy**: `serve lerobot` refuses policies with `n_obs_steps != 1` (SmolVLA, pi0, ACT are
  fine, diffusion policy is not).
* **LeRobot is pinned to 0.6.1** (APIs moved a lot between 0.4 and 0.6); the base model needs internet once (HF
  cache volume `hf-cache`, set `HF_TOKEN` against rate limits).
* **The SmolVLA base expects cameras named `camera1..3`**: `train` renames ours, and the serve command must pass the same
  `--rename` flags (it is printed for you). Forgetting them is the first thing to check if a served checkpoint behaves
  strangely.
* The PhysX error "Prim to replicate not found in USD stage" at start-up appears in every run and is harmless as far as
  I could tell; I did not find its cause.
* In this shell `ls` is aliased and prints `.` and `..`, `rm` may move to a trash: do not count directory entries with it.
* The ZED mount geometry was exported mirrored; `scripts/build_assets.py` reflects it (`MIRROR_CAD_Y`), and the mount pose
  on the hand (`MOUNT_POS/ROT` in `pick_place_env_cfg.py`) was tuned by hand in Isaac Lab by the project owner. If the
  real mount looks different, revisit both.
* The robot base frame is the env frame (the robot sits at each env's origin), the table top is at z = 0, "left" in task
  prompts means the robot's left (+y).

## 9. Decisions worth knowing

* **One shared policy server, batched requests**: once calls are batched, rendering costs more than inference (16 envs:
  about 206 ms of simulation per control step, against about 46 ms per step of amortised model time, since one 367 ms
  call serves 8 steps), so the number of envs per GPU is limited by rendering and GPU memory.
* **Failures only, but all outcomes counted**: `stats.jsonl` keeps every episode (success rates are the weakness signal),
  `manifest.jsonl` lists what is on disk and is written only once an episode is complete, so a crash cannot leave
  dangling entries.
* **Frames are buffered in RAM and encoded only for kept episodes**; keep an eye on RAM with many envs and long
  episodes (about 2 MB per env and step for 3 cameras).
* Actions are absolute joint position targets (7) + a binary gripper command at 15 Hz, the DROID `joint_pos` space;
  SmolVLA pads its 32-dim action space, which is why 8 dimensions work without changing the model.

## 10. Questions to settle with the project owner

* What success band and what failure quotas per task and tag should the adaptive loop aim for?
* Which Cosmos 3 model and interface will be the analyst / teacher, and where does it run (the H100 server has no RT
  cores, so it cannot host the simulator)?
* Is the DROID-style pi0.5 zero-shot baseline enough as a second policy, or should it be fine-tuned too?
