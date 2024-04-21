# MetodyStochastyczne2024

## TODO

- Implement the hms optimizer
  - Parsing kernel to extract hyperparameters
  - Implement the optimization loop with hms
  - Loading optimzed hyperparameters to the kernel and fixing it with `source_code.utils.fix_kernel(kernel)`

## Quick start

### Install dependencies

Run rhe following command from the project root to install the required dependencies:
```bash
pip install -e .
```

### Run the example
```bash
python source_code/gpr.py -k kernels/sample_kernel.py -d data/co2_clean.csv --column CO2 -v 
```
