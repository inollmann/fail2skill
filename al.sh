#!/usr/bin/env bash
# Entry point for the panda_cosmos_al containers. Run `./al.sh help`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$ROOT/docker"

# settings: docker/.env, then docker/.env.local
set -a
# shellcheck disable=SC1091
source "$DOCKER_DIR/.env"
# shellcheck disable=SC1091
[[ -f "$DOCKER_DIR/.env.local" ]] && source "$DOCKER_DIR/.env.local"
set +a
[[ "$ISAACLAB_PATH" = /* ]] || ISAACLAB_PATH="$ROOT/$ISAACLAB_PATH"
[[ "$AL_DATA" = /* ]] || AL_DATA="$DOCKER_DIR/$AL_DATA"
export AL_DATA

compose() {
    local env_files=(--env-file "$DOCKER_DIR/.env")
    [[ -f "$DOCKER_DIR/.env.local" ]] && env_files+=(--env-file "$DOCKER_DIR/.env.local")
    docker compose -f "$DOCKER_DIR/docker-compose.yaml" "${env_files[@]}" "$@"
}

prepare_runtime() {
    mkdir -p "$AL_DATA"
    # X11 access through a dedicated cookie file (no `xhost +`), so GUIs work from the containers
    local xauth_file="$DOCKER_DIR/.panda_al.xauth"
    : > "$xauth_file"
    if [[ -n "${DISPLAY:-}" ]] && command -v xauth >/dev/null; then
        xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$xauth_file" nmerge - 2>/dev/null || true
    fi
    chmod 644 "$xauth_file"
}

setup() {
    command -v docker >/dev/null || { echo "docker is not installed"; exit 1; }
    docker info 2>/dev/null | grep -qi nvidia || echo "[warn] NVIDIA container runtime not detected (install nvidia-container-toolkit)"
    if [[ ! -d "$ISAACLAB_PATH/.git" ]]; then
        echo "[setup] cloning Isaac Lab into $ISAACLAB_PATH"
        git clone "$ISAACLAB_REPO" "$ISAACLAB_PATH"
    fi
    local head
    head="$(git -C "$ISAACLAB_PATH" rev-parse HEAD)"
    if [[ "$head" != "$ISAACLAB_REF"* ]]; then
        if [[ -n "$(git -C "$ISAACLAB_PATH" status --porcelain)" ]]; then
            echo "[warn] $ISAACLAB_PATH is at ${head:0:9} with local changes; expected $ISAACLAB_REF. Not touching it."
        else
            echo "[setup] checking out Isaac Lab $ISAACLAB_REF"
            git -C "$ISAACLAB_PATH" fetch --quiet origin
            git -C "$ISAACLAB_PATH" checkout --quiet "$ISAACLAB_REF"
        fi
    fi
}

build_base() {
    setup
    echo "[build] isaac-lab-base from $ISAACLAB_PATH"
    touch "$ISAACLAB_PATH/docker/.isaac-lab-docker-history"
    (cd "$ISAACLAB_PATH/docker" && docker compose --file docker-compose.yaml --profile base --env-file .env.base build isaac-lab-base)
}

usage() {
    cat <<'EOF'
Usage: ./al.sh <command> [args]

Setup
  setup                 clone / check out the pinned Isaac Lab version
  build [sim|ros|policy]  build the images (default: sim + ros; sim also builds isaac-lab-base)

Data collection
  teleop [--gui]        MoveIt teleop: starts the simulator + MoveIt/RViz and opens the episode console
                        (--gui also opens the Isaac Sim window; RViz is always shown)
  generate [args]       scripted episodes, e.g. ./al.sh generate --num_envs 8 --num_episodes 200
  replay [args]         replay episodes / validate actions, e.g. ./al.sh replay --episodes data/episodes/x/failure/*
  export-lerobot [args] episodes -> LeRobotDataset for fine-tuning, e.g.
                        ./al.sh export-lerobot --source data/episodes/demos --out data/lerobot/demos_v0
  train [args]          fine-tune SmolVLA on such a dataset (LeRobot), e.g.
                        ./al.sh train --dataset data/lerobot/demos_v0 --name smolvla_iter0 --steps 20000
  report <run dirs>     success rates per task / failure tags from stats.jsonl of collection runs
  collect <campaign>    unattended failure collection with a served policy (configs/campaigns/*.yaml);
                        restarts the simulator after crashes and resumes from the files on disk
  registry              validate assets/objects and configs/tasks and list what they define
  policy [args]         roll out a policy served over websocket, e.g. ./al.sh policy --adapter droid --num_episodes 20
  serve [args]          policy server (LeRobot image), e.g. ./al.sh serve lerobot --checkpoint <dir|repo>
                        or ./al.sh serve hold (smoke test). Paths below data/ are /workspace/data/...

Tools
  shell                 interactive shell in the sim container
  ros-shell             interactive shell in the ROS container
  run <cmd...>          run a command in the sim container (e.g. ./al.sh run isaaclab -p scripts/...)
  test                  run the unit tests (simulator image); test-policy: LeRobot-dependent ones (policy image)
  logs                  follow the teleop container logs
  down                  stop all containers of this project

Settings live in docker/.env; put per-machine overrides (AL_GPU, AL_BRIDGE_PORT, ROS_DOMAIN_ID ...)
in docker/.env.local. Episodes are written to $AL_DATA (default: ./data).
EOF
}

# paths below the repo's data/ dir are translated to the container mount point
to_container_paths() {
    local arg abs data out=()
    data="$(realpath "$AL_DATA")"
    for arg in "$@"; do
        abs="$(realpath -m -- "$arg")"
        if [[ "$abs" == "$data"* && ( -e "$arg" || "$arg" == */* ) ]]; then
            out+=("/workspace/data${abs#"$data"}")
        else
            out+=("$arg")
        fi
    done
    printf '%s\n' "${out[@]}"
}

cmd="${1:-help}"
shift || true
case "$cmd" in
    setup) setup ;;
    build)
        target="${1:-all}"
        prepare_runtime
        if [[ "$target" == all || "$target" == sim ]]; then build_base; compose --profile sim build sim; fi
        if [[ "$target" == all || "$target" == ros ]]; then compose --profile teleop build moveit; fi
        if [[ "$target" == policy ]]; then compose --profile policy build policy; fi
        ;;
    teleop)
        prepare_runtime
        AL_SIM_VIEW=""
        [[ "${1:-}" == "--gui" ]] && AL_SIM_VIEW="--viz kit"
        export AL_SIM_VIEW
        compose --profile teleop up -d teleop-sim moveit
        trap 'compose --profile teleop stop teleop-sim moveit' EXIT
        echo "[teleop] simulator and MoveIt starting (./al.sh logs to follow). Episodes -> $AL_DATA/episodes/teleop"
        compose --profile teleop exec moveit /al_entrypoint.sh ros2 run al_ros_bridge al_episode
        ;;
    generate)
        prepare_runtime
        compose --profile sim run --rm sim isaaclab -p scripts/generate_episodes.py "$@"
        ;;
    replay)
        prepare_runtime
        mapfile -t args < <(to_container_paths "$@")
        compose --profile sim run --rm sim isaaclab -p scripts/replay_episode.py "${args[@]}"
        ;;
    collect)
        prepare_runtime
        [[ $# -ge 1 ]] || { echo "usage: ./al.sh collect configs/campaigns/<campaign>.yaml"; exit 2; }
        tty_flag=(); [[ -t 0 ]] || tty_flag=(-T)   # unattended (nohup, ssh without tty)
        # the isaaclab launcher turns every exit code into 1, so scripts/collect.py reports its status in a file
        status_name=".collect_status_$$"
        trap 'rm -f "$AL_DATA/$status_name"' EXIT
        restarts=0
        while true; do
            rm -f "$AL_DATA/$status_name"
            set +e
            compose --profile sim run --rm "${tty_flag[@]}" -e AL_COLLECT_STATUS_FILE="/workspace/data/$status_name" \
                sim isaaclab -p scripts/collect.py "$@"
            set -e
            code=1; [[ -f "$AL_DATA/$status_name" ]] && code="$(<"$AL_DATA/$status_name")"   # no file: crash
            [[ $code -eq 0 || $code -eq 2 ]] && exit "$code"
            restarts=$((restarts + 1))
            if [[ $restarts -gt ${AL_COLLECT_MAX_RESTARTS:-20} ]]; then echo "[collect] too many restarts"; exit "$code"; fi
            echo "[collect] exited with code $code, restarting in 30 s ($restarts/${AL_COLLECT_MAX_RESTARTS:-20})"
            sleep 30
        done
        ;;
    export-lerobot)
        prepare_runtime
        mapfile -t args < <(to_container_paths "$@")
        compose --profile policy run --rm --entrypoint python policy -m panda_al.export.lerobot_export "${args[@]}"
        ;;
    train)
        prepare_runtime
        mapfile -t args < <(to_container_paths "$@")
        compose --profile policy run --rm --entrypoint python policy -m panda_al.train "${args[@]}"
        ;;
    report) python3 "$ROOT/scripts/report_stats.py" "$@" ;;
    registry) prepare_runtime; compose --profile sim run --rm sim python scripts/check_registry.py "$@" ;;
    policy)
        prepare_runtime
        compose --profile sim run --rm sim isaaclab -p scripts/run_policy.py "$@"
        ;;
    serve)
        prepare_runtime
        compose --profile policy run --rm policy "$@"
        ;;
    shell) prepare_runtime; compose --profile sim run --rm sim bash ;;
    ros-shell) prepare_runtime; compose --profile teleop run --rm moveit bash ;;
    run) prepare_runtime; compose --profile sim run --rm sim "$@" ;;
    test) prepare_runtime; compose --profile sim run --rm sim python -m pytest -q tests ;;
    test-policy) prepare_runtime; compose --profile policy run --rm --entrypoint python policy -m pytest -q tests/test_export.py tests/test_policy.py ;;
    logs) compose --profile teleop logs -f ;;
    down) compose --profile sim --profile teleop down ;;
    help|-h|--help) usage ;;
    *) usage; exit 1 ;;
esac
