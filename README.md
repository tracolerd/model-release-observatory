# Model Release Observatory

**Make model experiments, evaluations, and releases traceable.**

Model Release Observatory is a planned MLOps learning project for tracking model experiments, artifact versions, evaluation results, and selected signals that may indicate data or performance drift. The aim is reproducibility and transparent release decisions, not a claim that one metric can guarantee model quality.

> **Project status:** Initial planning and repository setup. No model pipeline, tracking server configuration, registry workflow, or drift monitor has been implemented yet.

## Problem

A model artifact is difficult to trust without knowing how it was trained, which dataset was used, what configuration produced it, how it was evaluated, and whether a candidate release performs differently from a baseline.

Model Release Observatory is intended to connect these records into a small, reproducible workflow that can be run locally and later extended.

## Planned MVP

- Run a small, resource-friendly baseline model on a documented dataset.
- Record experiment parameters, metrics, code/configuration references, and artifact metadata.
- Track dataset fingerprints and versions without assuming a hash proves data quality.
- Compare candidate evaluation results with a baseline using a defined protocol.
- Store model artifacts and version metadata.
- Track a release state such as candidate, approved, or retired.
- Measure inference latency in a repeatable test harness.
- Add a basic data-distribution drift check with explicit limitations.
- Generate a release report that lists evidence, metrics, and known caveats.

## Proposed architecture

~~~mermaid
flowchart TD
    A[Versioned Dataset and Config] --> B[Training or Evaluation Run]
    B --> C[MLflow Tracking]
    B --> D[Metrics and Artifacts]
    D --> E[Baseline Comparison]
    E --> F[Release Review]
    F --> G[Model Registry and Version]
    H[New Evaluation Data] --> I[Drift Checks]
    I --> J[Monitoring Report]
~~~

Training, evaluation, metadata storage, and serving should be separate concerns. The first release should prioritize reproducibility and evidence before introducing multiple deployment environments.

## Proposed technology

| Area | Planned choice |
|---|---|
| Core language | Python |
| Experiment tracking | MLflow |
| Initial model | Small scikit-learn model suitable for local execution |
| API | FastAPI, after the core workflow is repeatable |
| Metadata | MLflow backend; PostgreSQL may be added for shared deployment |
| Optional serving | BentoML or another documented serving layer |
| Optional experiment platform | Weights & Biases adapter |
| Advanced experiment | PEFT/LoRA/QLoRA only when hardware and dataset size are appropriate |

External services should remain optional. A working local mode should not require cloud credentials or a paid inference endpoint.

## Reproducibility principles

- Record dataset identity, relevant preprocessing, model parameters, random seeds where meaningful, and dependency versions.
- Keep training and evaluation datasets separate.
- Define metrics before comparing runs.
- Store the evaluation configuration with the result.
- Distinguish model artifact version from application release version.
- Keep model artifacts and large datasets out of Git when appropriate; document how to obtain or regenerate them.
- Record when exact reproducibility is not guaranteed.

## Drift is not the same as model failure

A distribution shift can be a reason to investigate, but it does not automatically mean model quality has degraded. Conversely, a model may fail even when a simple drift detector does not fire. Drift reports should identify the evaluated features, baseline window, thresholds, sample size, and known limitations.

Metrics must fit the chosen task. Depending on the model, accuracy alone may be insufficient; precision, recall, F1, calibration, or task-specific metrics may be more informative.

## Development roadmap

1. Select a small dataset and define an evaluation protocol.
2. Implement a repeatable training/evaluation script and baseline tests.
3. Configure local MLflow experiment tracking.
4. Record artifact lineage and produce a comparison report.
5. Add a small FastAPI interface if it improves the workflow.
6. Define and test a transparent drift-check method.
7. Add release review states and model registry integration.
8. Consider BentoML serving, W&B integration, or a small fine-tuning experiment as separate extensions.
9. Add CI for linting, unit tests, and a lightweight deterministic evaluation subset.

## Testing strategy

- Dataset validation and schema checks.
- Repeated-run behavior under controlled seeds.
- Metric calculations against known examples.
- Tests for train/evaluation split separation.
- Artifact metadata and lineage checks.
- Baseline comparison and threshold boundary cases.
- Drift checks using synthetic distribution shifts and unchanged samples.
- API tests, if an API is introduced.
- Resource limits to keep the default workflow usable on a standard development machine.

## Security and governance

Do not commit licensed datasets, personal data, model credentials, or large private artifacts. Record dataset source and license. Validate uploaded metadata, restrict artifact access if a shared deployment is later added, and document the privacy implications of experiment logs. Model reports should not be presented as certification of production suitability.

## Current status

This repository currently contains the project brief and MIT license. No benchmark, model training result, deployment, fine-tuning result, or drift-detection accuracy claim is available yet.

## License

MIT. See [LICENSE](LICENSE).
