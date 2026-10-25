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
  policy [args]         roll out a policy served over websocket, e.g. ./al.sh policy --adapter droid --num_episodes 20
  serve [args]          policy server (LeRobot image), e.g. ./al.sh serve lerobot --checkpoint <dir|repo>
                        or ./al.sh serve hold (smoke test). Paths below data/ are /workspace/data/...

Tools
  shell                 interactive shell in the sim container
  ros-shell             interactive shell in the ROS container
  run <cmd...>          run a command in the sim container (e.g. ./al.sh run isaaclab -p scripts/...)
  test                  run the unit tests
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
        if [[ -e "$arg" && "$abs" == "$data"* ]]; then
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
    logs) compose --profile teleop logs -f ;;
    down) compose --profile sim --profile teleop down ;;
    help|-h|--help) usage ;;
    *) usage; exit 1 ;;
esac
