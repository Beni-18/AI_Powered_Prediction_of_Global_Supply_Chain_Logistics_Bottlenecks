<!-- PR title must follow the commit format:  <type>(<stage>): <message>   e.g.  ml(stage01): xgboost baseline + calibration -->
<!-- Do NOT add "Generated with ..." or any AI/tool attribution to this description. -->

## What this PR delivers
<!-- One or two plain sentences. -->

## Role and stage
- Role: <!-- 1 ML / 2 EDA / 3 Data+Integration / 4 Evaluation -->
- Stage / day: <!-- e.g. Stage01_ML, Day 1 -->

## Hand-off
- [ ] My `Stage0N/HANDOFF/0K_*.md` file is updated (status, files produced, decisions)
- [ ] I ran `python scripts/validate_contract.py <schema> <file>` on every contract file I produced and pasted the output in the hand-off file
- [ ] Next role(s) who need this: <!-- e.g. Role 4 can now evaluate -->

## Checks
- [ ] Only files in my lane changed (or the commit has a `Cross-role:` line explaining who agreed)
- [ ] Every code step has a plain-English comment (docs/08_COMMENTING_STANDARD.md)
- [ ] No absolute paths, no secrets, no files over 50 MB
- [ ] Every number in my reports is produced by code, not typed by hand
- [ ] `python -m pytest tests -q` passes locally
