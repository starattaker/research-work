# Checkpoint — paper expanded; v6 ICC production (2026-09-04)

**Status:** Paper draft expanded. Friend: one command to finalize ICC + figures + push.

## Intersection + ICC (important)

Severity **requires** CEJ + **intersection** + apex. You cannot remove intersection from ICC.

**Use v6 intersection** (`runs/keypoints/v6_intersection/best.pt`, OKS 0.894).  
**Do NOT use v7 intersection** (OKS 0.882) for ICC.

## Friend GPU — ONE command

```bash
cd ~/faraz/Test_work/research-work && git fetch origin denpar-severity-replication && git pull origin denpar-severity-replication --no-rebase --no-edit && bash scripts/run_paper_finalize_friend.sh
```

## Build PDF (local)

```bash
bash scripts/build_paper_pdf.sh
```

## Green placeholders in paper

Fill after friend run: `axis_severity_icc.json`, figure PNGs in `paper/figures/`
