#!/bin/bash
# Boot-time auto-resume. The Aug 19 reboot killed the training run and NOTHING
# restarted it, so the machine idled for 12.9 days before anyone noticed. That
# cost more than either crash did. This is installed via `crontab @reboot` and
# makes an interruption cost only the fold that was in flight.
REPO=/media/inartrans2/6c8f6887-9acf-4794-b990-8de964c59e871/rhl/cookies
LOG="$REPO/_autoresume.log"

# The repo lives on an external mount that may not be ready at boot.
for i in $(seq 1 60); do
  [ -d "$REPO/artifacts/deep_iekf_online" ] && break
  sleep 10
done
[ -d "$REPO/artifacts/deep_iekf_online" ] || { echo "$(date): repo mount never appeared" >> "$LOG"; exit 1; }

cd "$REPO" || exit 1
n_done=$(ls artifacts/deep_iekf_online/.deep_iekf_fold_*.done 2>/dev/null | wc -l)
if [ "$n_done" -eq 7 ]; then
  echo "$(date): all 7 folds complete, nothing to resume" >> "$LOG"; exit 0
fi
# Match the trainer's distinctive argv, not the bare filename: `pgrep -f` scans
# whole command lines, so a script or shell that merely mentions the filename
# would otherwise look like a running trainer and suppress the resume forever.
if pgrep -f "train_ai_imu\.py --mode loo" >/dev/null; then
  echo "$(date): training already running, not starting a second copy" >> "$LOG"; exit 0
fi
echo "$(date): resuming at ${n_done}/7 folds complete" >> "$LOG"
docker start cookies_build >/dev/null 2>&1
nohup ./_resume_pipeline.sh >> _resume.log 2>&1 &
