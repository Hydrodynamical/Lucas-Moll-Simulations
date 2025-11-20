# Lucas–Moll Simulation Setup

Follow the steps below to create and activate a dedicated `.venv` for this project on macOS or Linux shells such as `zsh`.

## 1. Create the Virtual Environment

```bash
python3 -m venv .venv
```

The command builds a local `.venv/` directory containing an isolated Python interpreter and site-packages directory.

## 2. Activate the Environment

```bash
source .venv/bin/activate
```

After activation the shell prompt gains a `(.venv)` prefix, and `python`/`pip` commands now resolve inside the virtual environment.

## 3. Install Project Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The first line ensures `pip` is up to date; the second installs everything listed in `requirements.txt` (NumPy, Matplotlib, Pandas).

## 4. Run the Simulation

```bash
python v1.py
```

This executes the Lucas–Moll jump–diffusion simulation using the packages from the active `.venv`.

## 5. Deactivate When Finished

```bash
deactivate
```

Leaving the environment restores the shell to the system-wide Python interpreter.

## 6. Versions

- v1.py is the initial many agent attempt with incorrect learning function
- v2.py is the updated many agent code with incorrect learning function
- v3.py is the PDE simulator with the correct learning function
- v4.py is the PDE simulator with the correct learning function, and produces the an image of the transition to the lottery society below $\alpha_1 \leq 1$.
- v5.py is the corrected many agent code.
- v6.py overlays the PDE simulator against the particle simulator. The learning front for the particle simulator lags, although the median fronts are the same speed. This indicates the presence of a finite $N$ effect.
