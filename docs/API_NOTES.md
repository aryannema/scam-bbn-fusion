# Library API notes

Short record of library details we checked against the *pinned* versions, so later phases do
not rediscover them and so we can answer "why did you write it that way?" in the viva.

## pgmpy 1.1.2

Verified by running the code in this venv, not from memory.

| What we need | The call in pgmpy 1.1.2 | Note |
|---|---|---|
| The network class | `from pgmpy.models import DiscreteBayesianNetwork` | Renamed in 1.x. In older tutorials (0.1.x) it is `BayesianNetwork`; that name is why we pin the version. |
| Learn CPTs | `from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator` | `BayesianEstimator(..., prior_type="BDeu")` for the smoothed version. |
| Inference | `from pgmpy.inference import VariableElimination` | Used by F1 (scoring), F7 (counterfactual search). |
| Structure learning | `from pgmpy.estimators import HillClimbSearch` | — |
| Scoring function | `est.estimate(scoring_method="bic-d")`, or `"bdeu"` | Passed as a **string**. Valid: `k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g, ll-cg, aic-cg, bic-cg`. |
| Max parents per node | `est.estimate(..., max_indegree=3)` | As the brief requires. |
| **Forbidding edges (the "black list")** | `from pgmpy.estimators import ExpertKnowledge`, then `est.estimate(expert_knowledge=ExpertKnowledge(forbidden_edges=[...]))` | **Changed in 1.x.** The old `black_list=` / `white_list=` arguments are gone. `ExpertKnowledge` takes `forbidden_edges`, `required_edges`, `temporal_order`, `search_space`. |
| Deprecation to avoid | `pgmpy.estimators.StructureScore` emits a `FutureWarning` (removed in v1.3.0) | If we need a score object rather than a string, import from `pgmpy.structure_score` (`BIC`, `BDeu`, `K2`, `LogLikelihood`, ...). |

## Other pinned versions worth knowing

- **pandas 3.0.6** — major version 3. Copy-on-write is the default, so chained assignment
  (`df[mask]['col'] = x`) silently does nothing. We always assign with `.loc[mask, "col"] = x`.
- **numpy 2.5.4** — numpy 2.x API. `np.float_` and other 1.x aliases no longer exist.
- **torch is NOT installed.** pgmpy 1.x declares torch only as an optional extra (`pgmpy[torch]`),
  so our install stays around 850 MB instead of several gigabytes. Nothing we use needs it.
