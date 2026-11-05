#!/bin/bash
# Job-started hook of the self-hosted runner (ACTIONS_RUNNER_HOOK_JOB_STARTED, set by
# runner_batch.sh). The runner runs under a personal cluster account and is started
# for ONE workflow run (CI_RUN_ID): any other job (another run, a branch, a fork PR)
# is refused. A non-zero exit fails the job before any of its steps runs.
# ceci-gate copies it to the cluster at every start, from the vmadgraph VM
# (/usr/local/share/ceci-gate): install it there again after a change.
if [ "$GITHUB_REPOSITORY" = "MadGraphTeam/MadGraph7" ] && [ -n "$CI_RUN_ID" ] && [ "$GITHUB_RUN_ID" = "$CI_RUN_ID" ]; then
    echo "Job of workflow run $GITHUB_RUN_ID accepted"
    exit 0
fi
echo "::error::this runner only runs the jobs of workflow run ${CI_RUN_ID:-?} of MadGraphTeam/MadGraph7, not of run $GITHUB_RUN_ID of $GITHUB_REPOSITORY"
exit 1
