The `obstacle_cylinder` corruption mode now works end to end. An 8-episode run produced 8 failures, so the cylinder does not give you "sometimes" yet: it failed every episode. Replays of two of them reproduced the failure at the same step counts (29 and 87). Four normal episodes with no corruption all succeeded, and the 15 unit tests pass.

**How it works**
- **Obstacle:** a static cylinder, height 0.18 and radius = cube side / 2, defined in [pick_place_env_cfg.py](src/panda_al/tasks/pick_place_env_cfg.py). When no obstacle is in use it sits parked under the table, so other episodes are unaffected.
- **Placement:** the expert places it in `reset()` ([scripted.py](src/panda_al/scripted.py)), right after the env reset has randomized the cubes. It lands between the cubes at 0.4–0.6 of the way from the pick cube to the place cube, with up to 5 cm of sideways offset. The expert plan is untouched.
- **What gets recorded:** `meta.json` stores the radius, height, along fraction, lateral offset and the resulting position (robot base frame) under `extra.corruption`. The obstacle is also saved in `initial_state.npz`, which is why replays include it. I added `dashcam.snapshot_initial_state()` and call it from `generate_episodes.py` after the expert places the cylinder.
- **Collision handling:** a contact sensor on the cylinder triggers a new `obstacle_collision` termination at any touch over 1 N. The episode ends immediately and is filed as a failure. The term name is in `termination_terms`.

Try it with `./al.sh generate --modes obstacle_cylinder --corrupt_ratio 1 --num_episodes 20 --run_name obstacle`. I also added a README section and `tests/test_scripted.py`.

**Why every episode failed**
- The cylinder sits on the straight line the hand travels, and the hand is 0.2 m wide. At 5 cm lateral offset it still collides, so every episode collided.
- I looked at two failures. In one, the hand hit the cylinder while approaching the pick cube, since the cubes can be only 12 cm apart. In the other, the hand carrying the red cube hit it in transit. Of the 8 failures, 4 ended within about 2 s of the start and 4 later in the run.
- If you want a mix of hits and misses, I can shrink the along fraction (for example to 0.5 ± 0.1) or raise the lateral offset to 0.10–0.15 m. I left it as is since you want to gather failures first.

The cylinder is a more saturated orange now, because the first color washed out under the scene lighting.