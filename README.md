# When Is a Task Library Enough? Coverage and In-Context Forecasting

Code, preregistrations, and results for the paper of the same name (NeurIPS 2026 FMTS workshop).

**Paper (camera-ready PDF):** [paper.pdf](paper.pdf)

When a sequence model forecasts a system it has never seen, it might be estimating that system's dynamics from the context, or it might be retrieving a similar system it saw during training. Accuracy alone usually can't tell these apart. This project makes the question testable for a specific prompt by comparing the size of the model's training library, M, with the number of prior draws the posterior for that prompt would need, N2 = exp D2(posterior ‖ prior). The ratio R = M/N2 is the library's coverage. If R is far below 1, Bayesian lookup over the library cannot match continuous Bayes, so a model that does match it must be doing something other than lookup.

Main results:

- In a preregistered AR(1) experiment, assigning more coverage improves posterior approximation and multi-step forecasts, including against a shuffled control with the same raw budget.
- A Transformer trained on 4,096 coupled eight-dimensional linear systems comes within 0.1% of continuous-Bayes risk at the longest context, even though its library is 6.84 bits short of resolving the posterior. Its predictions sit much closer to continuous Bayes than to Bayes restricted to its library (S = 0.973). With 256 training systems the model still forecasts well (8.9% above Bayes risk), but its predictions sit almost exactly between the two references.
- On in-context linear regression, a Transformer comes within 5% of Bayes risk while a parameter-matched GRU on the same data stays 30% above it. Both have moved away from lookup.

All data is synthetic. Every experiment except the Transformer training has a CPU-only replay.

## Layout

Each directory is one experiment and holds its frozen preregistration, the exact source that produced the results, and the compact outputs reported in the paper.

| Directory | Contents |
|---|---|
| `temporal/` | AR(1) histories with exact N2, plus quadrature and direct-simulation checks |
| `coverage_intervention/` | the preregistered coverage intervention (Figure 2): worker, aggregator, shard metadata, summary, figures |
| `learned_temporal/` | the Transformer on coupled 8-D linear systems at M = 256 and 4,096 (Figure 3): training source and audit |
| `checkpoint/` | linear-regression Transformer checkpoints at M = 256 to 16,384: replay and audit |
| `seed_replication/` | two extra seeds at M = 4,096 under a frozen decision rule |
| `architecture_comparison/` | the parameter-matched GRU baseline and paired analysis (Table 1) |
| `negative_controls/` | preregistrations and result summaries for the functional-coverage and path-Jacobian controls. The code for these depends on an earlier, larger experiment suite and is not included. |

`MANIFEST.json` lists the SHA-256 and size of every file, including the model files described below.

## Model files

The trained model states, task libraries, and saved predictions (about 75 MB of `.pt` and `.npz` files) are attached to the [v1.0 release](https://github.com/i-habib/task-library-coverage/releases/tag/v1.0) instead of stored in git. Unpack them over a checkout:

```bash
curl -LO https://github.com/i-habib/task-library-coverage/releases/download/v1.0/model-states.zip
unzip model-states.zip
python verify_manifest.py
```

## Reproducing the results

The reference environment was Python 3.14 with NumPy 2.4.6, SciPy 1.18.0, PyTorch 2.13.0, and Matplotlib.

```bash
pip install -r requirements.txt

# AR(1) temporal study (add --smoke for a 20-second check)
python temporal/fmts_temporal.py --output temporal/replay-output

# learned coupled dynamics: recompute every reported statistic from the saved predictions
python learned_temporal/audit_fmts_learned_temporal.py \
  --m256-root learned_temporal --m4096-root learned_temporal --output learned_temporal/replay-audit

# linear-regression checkpoints: replay inference on CPU (use --pilot --count 64 --bootstrap-replicates 20 for a quick check)
cd checkpoint
python fmts_checkpoint_audit.py --use-extracted --output-root artifacts/fmts-checkpoints --full --device cpu
```

For the coverage intervention, the four raw row files are not included. Run the worker with `--n 16`, `--n 64`, `--n 256`, and `--n 1024`, put the outputs in one directory, and run `aggregate_fmts_coverage_intervention.py --aggregate`. The architecture comparison's paired analysis for all four context lengths is in `architecture_comparison/r1_audit.json`.

The 0/3, 2/3, and 3/3 seed counts in the paper are descriptive. Bootstrap intervals cover variation over test contexts, not over training seeds.

## Acknowledgments

The linear-regression Transformer code in `checkpoint/`, `seed_replication/`, and `architecture_comparison/` adapts the code released with Raventós, Paul, Chen, and Ganguli, "Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression" (NeurIPS 2023), pinned to commit `7c35803e68c49f94ac2e618ce4e1bc14470b771b`. That code is Apache-2.0 licensed.

## Citation

```bibtex
@inproceedings{habib2026tasklibrary,
  title     = {When Is a Task Library Enough? Coverage and In-Context Forecasting},
  author    = {Habib, Ivan},
  booktitle = {NeurIPS 2026 FMTS Workshop},
  year      = {2026}
}
```

## License

MIT, except for the adapted Raventós et al. code noted above, which remains under Apache-2.0.
