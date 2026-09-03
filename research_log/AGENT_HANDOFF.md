# Agent handoff

Read: **CHECKPOINT.md** → **07_severity_icc.md** → **08_axis_severity.md** → **paper/replication_progress.tex**

**ICC weights (production):** all `v6_*` — especially `v6_intersection` (not v7).

**Friend one command** (if merge conflict: abort + prefer GitHub):
```bash
cd ~/faraz/Test_work/research-work && (git merge --abort 2>/dev/null || true) && bash scripts/run_paper_finalize_friend.sh
```

**Git divergent branches fix:** always `git pull --no-rebase` (in `sync_friend_repo.sh`).
