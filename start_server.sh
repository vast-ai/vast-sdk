#!/bin/bash

set -e -o pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

SERVER_DIR="$WORKSPACE_DIR/vast-pyworker"
ENV_PATH="$WORKSPACE_DIR/worker-env"
DEBUG_LOG="$WORKSPACE_DIR/debug.log"
PYWORKER_LOG="$WORKSPACE_DIR/pyworker.log"

REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"
USE_SSL="${USE_SSL:-true}"
WORKER_PORT="${WORKER_PORT:-3000}"
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# make all output go to $DEBUG_LOG and stdout without having to add `... | tee -a $DEBUG_LOG` to every command
exec &> >(tee -a "$DEBUG_LOG")

function echo_var(){
    echo "$1: ${!1}"
}


echo "start_server.sh"
date

echo_var REPORT_ADDR
echo_var WORKER_PORT
echo_var WORKSPACE_DIR
echo_var SERVER_DIR
echo_var ENV_PATH
echo_var DEBUG_LOG
echo_var PYWORKER_LOG
echo_var WORKER_SDK
echo_var PYWORKER_REPO
echo_var PYWORKER_REF

# Populate /etc/environment with quoted values
if ! grep -q "VAST" /etc/environment; then
    env -0 | grep -zEv "^(HOME=|SHLVL=)|CONDA" | while IFS= read -r -d '' line; do
            name=${line%%=*}
            value=${line#*=}
            printf '%s="%s"\n' "$name" "$value"
        done > /etc/environment
fi

setup_env() {
    echo "setting up venv"

    # Ensure uv is installed
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        [[ -f ~/.local/bin/env ]] && source ~/.local/bin/env
    fi

    # Fork testing
    [[ ! -d "$SERVER_DIR" ]] && git clone "${PYWORKER_REPO:-https://github.com/vast-ai/pyworker}" "$SERVER_DIR"
    if [[ -n ${PYWORKER_REF:-} ]]; then
        (cd "$SERVER_DIR" && git checkout "$PYWORKER_REF")
    fi

    # (Re)create venv
    uv venv --python-preference only-managed "$ENV_PATH" -p 3.10

    # Activate the newly created venv
    # shellcheck disable=SC1090
    source "$ENV_PATH/bin/activate"

    # Install requirements if present
    if [ -f "${SERVER_DIR}/requirements.txt" ]; then
        uv pip install -r "${SERVER_DIR}/requirements.txt"
    fi

    touch ~/.no_auto_tmux
}

# Decide if we actually have a usable venv
NEED_ENV_SETUP=false

# Missing directory, or clearly broken / incomplete venv
if [ ! -d "$ENV_PATH" ] \
   || [ ! -x "$ENV_PATH/bin/python" ] \
   || [ ! -f "$ENV_PATH/bin/activate" ]; then
    NEED_ENV_SETUP=true
fi

# If we don't have the server checkout yet, treat as needing setup as well
if [ ! -d "$SERVER_DIR" ]; then
    NEED_ENV_SETUP=true
fi

if [ "$NEED_ENV_SETUP" = true ]; then
    setup_env
else
    # uv installer may have dropped this file; source it if present
    [[ -f ~/.local/bin/env ]] && source ~/.local/bin/env

    # Activate existing venv (use ENV_PATH, not WORKSPACE_DIR/worker-env)
    # shellcheck disable=SC1090
    source "$ENV_PATH/bin/activate"

    echo "environment activated"
    echo "venv: $VIRTUAL_ENV"
fi

if [ "${WORKER_SDK:-false}" = true ]; then
    echo "Using Vast.ai SDK"
    uv pip install git+https://github.com/vast-ai/vast-sdk.git@remote


if [ "$USE_SSL" = true ]; then

    cat << EOF > /etc/openssl-san.cnf
    [req]
    default_bits       = 2048
    distinguished_name = req_distinguished_name
    req_extensions     = v3_req

    [req_distinguished_name]
    countryName         = US
    stateOrProvinceName = CA
    organizationName    = Vast.ai Inc.
    commonName          = vast.ai

    [v3_req]
    basicConstraints = CA:FALSE
    keyUsage         = nonRepudiation, digitalSignature, keyEncipherment
    subjectAltName   = @alt_names

    [alt_names]
    IP.1   = 0.0.0.0
EOF

    openssl req -newkey rsa:2048 -subj "/C=US/ST=CA/CN=pyworker.vast.ai/" \
        -nodes \
        -sha256 \
        -keyout /etc/instance.key \
        -out /etc/instance.csr \
        -config /etc/openssl-san.cnf

    curl --header 'Content-Type: application/octet-stream' \
        --data-binary @//etc/instance.csr \
        -X \
        POST "https://console.vast.ai/api/v0/sign_cert/?instance_id=$CONTAINER_ID" > /etc/instance.crt;
fi


export REPORT_ADDR WORKER_PORT USE_SSL UNSECURED

if [ "${WORKER_SDK:-false}" = true ]; then
    [ ! -f "$SERVER_DIR/worker.py" ] && echo "worker.py not found!" && exit 1
    WORKER_PATH="worker"
else
    [ ! -d "$SERVER_DIR/workers/$BACKEND" ] && echo "$BACKEND not supported!" && exit 1
    WORKER_PATH="workers.$BACKEND.server"
fi


cd "$SERVER_DIR"
echo "launching PyWorker server at $WORKER_PATH"

python3 -m "$WORKER_PATH" |& tee -a "$PYWORKER_LOG"

