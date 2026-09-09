# Agent handoff

Read: **CHECKPOINT.md** → **07_severity_icc.md** → **08_axis_severity.md** → **paper/replication_progress.tex**

**Production ICC weights:** all `v6_*` — especially `v6_intersection` (OKS 0.894), never v7 (0.882).

**Headline result:** mask-PCA axis severity raises honest test ICC to **0.823** vs paper Eq.1 0.726 (identical pairing, v6, match_by_slot).

**Status (2026-09-05):** This run is COMPLETE.
- Axis ICC computed on all splits → `research_log/axis_severity_icc.json`
- 16 axis figures + result PNGs committed as `d141c67` (present on `origin/denpar-severity-replication`)
- Paper `replication_progress.tex` updated (tab:axis-icc filled, abstract, discussion, new "Why mask-PCA" subsection); PDF builds locally with MiKTeX/pdflatex
- v7 training logs (CEJ 0.928 / intersection 0.882 / apex 0.881) confirm v6 stays production for ICC

**Friend one command** (if merge conflict: abort + prefer GitHub):
```bash
cd ~/faraz/Test_work/research-work && (git merge --abort 2>/dev/null || true) && bash scripts/run_paper_finalize_friend.sh
```

**Now current one-liners for a fresh machine (pull + run):**
```bash
cd <repo> && git fetch origin denpar-severity-replication && git pull origin denpar-severity-replication --no-rebase --no-edit && bash scripts/run_paper_finalize_friend.sh
```
or, to only recompute the paper + figures (skip the GPU ICC sweep — weights already exist):
```bash
bash scripts/run_paper_resume_axis_friend.sh
```
(That script skips the ICC parameter sweep, recomputes axis ICC + figure PNGs, then stages+commits figures for push. Note: `git push` from the friend shell may need a real PAT/SSH — the GitHub password-style auth used in pasted logs failed on the friend's box. Push from the local Windows repo with working creds instead, or configure a PAT.)

**Local build (Windows, MiKTeX):**
```bash
cd paper && pdflatex -interaction=nonstopmode replication_progress.tex && bibtex replication_progress && pdflatex -interaction=nonstopmode replication_progress.tex && pdflatex -interaction=nonstopmode replication_progress.tex
```

**Git divergent branches fix:** always `git pull --no-rebase` (in `sync_friend_repo.sh`).
