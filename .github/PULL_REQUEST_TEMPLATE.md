## What

<!-- What does this change do, and why? One short paragraph. Say if the PR is stacked on another one. -->

## Results / Checked

<!-- Measured results (with the command and the report file they come from) or what you checked.
     Every number must be cross-checked against reports/*.json. Report negative results honestly. -->

## Limits

<!-- What this does not show or cover: one machine, a few seeds, proxies instead of real photos, drafts that need human review... -->

## Checklist

- [ ] `pytest -q` and `ruff check .` pass locally
- [ ] Every number in the README or this PR was cross-checked against `reports/*.json`
- [ ] No data (`data/`), weights (`*.pt`), paintings from WikiArt or third-party photos are committed (check `git diff --cached --stat`)
- [ ] The README change is additive and small, or this PR says why not
