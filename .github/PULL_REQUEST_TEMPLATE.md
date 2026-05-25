<!-- Read CONTRIBUTING.md before opening this PR. Maintainer review guide: maintainer/03-pr-types.md -->

## Summary

<!-- One sentence: what changed. -->

## Why

<!-- Link to a related issue with `Closes #N` (auto-closes on merge) or `Refs #N`. Add motivation if not obvious from the issue. -->

## Verification

<!-- How did you verify this works? E.g. `tools/bench.py` output, manual test, CI alone. If you could not verify end-to-end (e.g. baselines not landed yet), say so explicitly — the maintainer will decide whether code-inspection is sufficient. -->

## Checklist

- [ ] Read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] One logical change per PR — one of: new kernel / optimization / benchmark / tooling / docs / KB / hardware-tier / bug fix (see [`maintainer/03-pr-types.md`](../maintainer/03-pr-types.md))
- [ ] For **tooling** PRs (`tools/*.py`, `.github/workflows/*`): existing `key=value` output keys are unchanged
- [ ] For **kernel** PRs: `kernel_fn` does not call back to PyTorch ops (see [CONTRIBUTING.md § What kernel.py may not do](../CONTRIBUTING.md#what-kernelpy-may-not-do))
- [ ] For **benchmark** PRs: used the template from [`maintainer/templates/pr-benchmark-submission.md`](../maintainer/templates/pr-benchmark-submission.md) and attached `workspace/results.tsv` + `tools/prepare.py` output
- [ ] CI is green (or the PR fixes a CI bug — explain)
