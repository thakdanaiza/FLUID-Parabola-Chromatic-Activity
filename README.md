# FLUID-Parabola-Chromatic-Activity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23713125.svg)](https://doi.org/10.5281/zenodo.23713125)

Standalone command-line processing for the V06 flight selections. It writes the
raw chromatic-activity stack and the complete short- and long-interval metric
series. It does not prepare/stabilize video and does not create figures or paper
tables.

1. Put the prepared quantitative input at `input/tracked_full_frame_rectified.mp4`
   or change `input_video` in `config.yaml`.
2. Install the pinned packages in `requirements.txt`.
3. Run `python run_chromatic_activity.py --config config.yaml`.

The supplied config contains the exact V06 Window A and continuous pre-transient
Window B selections and algorithm parameters. Use `--profile window_b` to run a
single selection. CuPy/CUDA is required because the retained flight activity
stacks used the explicit GPU sRGB-D65-CIELAB implementation.

Each profile produces `activity_stack.h5`, `raw_activity_metrics.csv`,
`short_interval_metrics.csv`, `long_interval_metrics.csv`, and
`run_metadata.json`.

## Citation

If you use this software, cite the archived v1.0.1 release:

> Sirisombat, T. (2026). FLUID-Parabola Chromatic Activity (v1.0.1).
> Zenodo. https://doi.org/10.5281/zenodo.23719667

The version-independent DOI for this software is
https://doi.org/10.5281/zenodo.23713125. Citation metadata is also available in
`CITATION.cff`.

