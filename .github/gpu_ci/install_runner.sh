#!/bin/bash
# One-time installation of the GitHub self-hosted runner used by the GPU CI
# (.github/workflows/gpu_runner_ci.yml). Run it on the cluster login node, from a
# clone of MadGraph7 on your own machine, e.g. for manneback:
#
#   { echo "TOKEN=$(gh api -X POST repos/MadGraphTeam/MadGraph7/actions/runners/registration-token --jq .token)"
#     cat .github/gpu_ci/install_runner.sh; } | ssh manneback bash -l -s -- manneback-gpu
#
# (the registration token goes through stdin so that it never shows up in 'ps').
# The runner is installed in ~/mg7-gpu-ci/runner-LABEL and registered with the
# name and label LABEL. It is NOT started here: the CI starts it inside a Slurm
# job for the duration of one workflow run (see runner_batch.sh).
# Running the script again re-registers the runner, which is needed when GitHub
# removed it after 14 days without any connection.
set -eo pipefail
LABEL=${1:?missing runner label (e.g. manneback-gpu)}
REPO_URL=${2:-https://github.com/MadGraphTeam/MadGraph7}
: "${TOKEN:?TOKEN must contain a runner registration token}"
RUNNER_DIR=$HOME/mg7-gpu-ci/runner-$LABEL
WORK_DIR=${GLOBALSCRATCH:-$HOME}/mg7-gpu-ci/work-$LABEL

mkdir -p "$RUNNER_DIR" "$WORK_DIR"
cd "$RUNNER_DIR"
if [ ! -x config.sh ]; then
    VERSION=$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p')
    echo "Installing actions-runner $VERSION in $RUNNER_DIR"
    curl -fsSL -o runner.tar.gz "https://github.com/actions/runner/releases/download/v${VERSION}/actions-runner-linux-x64-${VERSION}.tar.gz"
    tar -xzf runner.tar.gz
    rm runner.tar.gz
fi
# forget a previous registration; --replace then takes over the runner with the same name
rm -f .runner .credentials .credentials_rsaparams
./config.sh --unattended --replace --url "$REPO_URL" --token "$TOKEN" \
            --name "$LABEL" --labels "$LABEL" --work "$WORK_DIR"
echo "Runner $LABEL registered in $RUNNER_DIR (jobs work in $WORK_DIR)"
