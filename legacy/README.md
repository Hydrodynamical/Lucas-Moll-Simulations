# Historical scripts

These files are preserved byte-for-byte from commit `3df9e2b` for research
provenance. They are not maintained entry points. Use the package in
[`../lucas_moll`](../lucas_moll) for new experiments.

| File | Historical role |
| --- | --- |
| `v1.py` | Initial particle experiment with the wrong learning functional for the intended model |
| `v2.py` | Revised particle experiment, still with that incorrect learning functional |
| `v3.py` | PDE solver with the intended learning functional |
| `v4.py` | PDE parameter sweep and front-speed plots |
| `v5.py` | Particle simulation with the corrected learning functional |
| `v6.py` | Particle/PDE comparison, including two-million-particle defaults |

Several scripts execute simulations or plot windows when imported. They can
write fixed filenames in the current directory. `v6.py` retains biased peer
selection, staggered PDE diagnostic times, and different fitting windows;
see [migration notes](../docs/MIGRATION.md). Run them only deliberately, in a
separate output directory, with dependencies from `pip install -e '.[legacy]'`
installed from the repository root. Their behavior has not been silently
changed to match the active runner.
