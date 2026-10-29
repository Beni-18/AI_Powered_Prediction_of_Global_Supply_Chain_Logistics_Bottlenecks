# Decisions log

Every team decision that changes scope, a contract, a threshold or a method is written down here so that nobody
re-argues it at the debate. One file per decision: `NNNN-short-title.md` (use the next free number; if two people
pick the same number, the later one adds a role suffix, e.g. `0007-eval.md`).

Template:

```
# NNNN - Title
Date: YYYY-MM-DD   Deciders: roles   Status: PROPOSED | ACCEPTED | SUPERSEDED by NNNN
## Context        (what forced the decision, in plain English)
## Options        (2-3 options with pros and cons)
## Decision       (what we chose)
## Consequences   (what changes for which role; what we give up)
```

Decisions already waiting for the team (create these files on Day 0 / Day 1):

| # | Topic | Owner | When |
|---|---|---|---|
| 0001 | Event catalogue for the simulator (frequencies, durations, precedents) | Role 2 | Day 0 |
| 0002 | Compute plan (CPU only vs GPU session for Day 4-5) | Role 1 | Day 0 |
| 0003 | Ratify `label_policy.json` thresholds (6 h / 24 h) | Role 3 + 4 | Day 1 Team Huddle |
| 0004 | Ratify `cost_policy.json` (miss = 10, false alarm = 1) and alert budget | Role 4 | Day 1 Team Huddle |
| 0005 | NLP cost matrix and review confidence threshold (0.70) | Role 4 | Day 3 |
| 0006 | Final SLM system and serialisation format | Role 1 + 2 | Day 4 |
| 0007 | Wildcard scenario parameters | Role 2 | Day 5 |
