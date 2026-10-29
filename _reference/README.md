# _reference/ - local-only reference material

Everything in this folder **except this README** is git-ignored. It exists so your IDE/assistant can read the
reference material, because most assistants can only read files inside the open workspace.

Put these two things here **on your own device** (do not commit them):

```
_reference/
  Supply_Chain_Bottleneck_Industry_Oriented_Project/     <- the already-built example project (unzipped)
  03 - AI_Powered_Prediction_of_Global_Supply_Chain_Logistics_Bottlenecks.pdf   <- the problem statement
```

## How to use the reference project

- Use it for **structure** only: five stage folders, five role scripts per stage (`01_data_engineer.py` ... `05_integration_engineer.py`), a `data/` folder per stage, a root `run_pipeline.py`, `app.py`, `tests/`.
- Do **not** copy its code, data or models into this repo. Its data is synthetic with labels derived from the features, so its near-perfect scores are an artefact; our project fixes exactly that (see `docs/00_PROJECT_BRIEF.md`, section "What the reference teaches us, and where we beat it").
- If you quote anything from it in a debate, say it is the reference example, not our work.
