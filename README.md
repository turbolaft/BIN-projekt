# Evolutionary MLP Search with Layer-Wise Quantization

This project trains a small MLP classifier on a small UCI-style dataset and uses an evolutionary algorithm to search for:

- the number of hidden layers,
- the number of neurons in each hidden layer,
- the quantization level used in each hidden layer.

The search optimizes a fitness score that balances classification accuracy with memory/computation efficiency.

## Project layout

- `main.py` - project entry point
- `mlp_evolution/config.py` - experiment configuration
- `mlp_evolution/data.py` - dataset loading and train/validation/test split
- `mlp_evolution/genotype.py` - genotype definition and genetic operators
- `mlp_evolution/quantization.py` - fake quantization for weights and activations
- `mlp_evolution/model.py` - quantized MLP phenotype
- `mlp_evolution/train.py` - training and evaluation helpers
- `mlp_evolution/fitness.py` - fitness computation
- `mlp_evolution/evolution.py` - evolutionary search loop

## Genotype

The genotype is encoded as:

```text
[num_hidden_layers, n1, n2, n3, q1, q2, q3]
```

With a fixed maximum number of hidden layers. Example:

```text
[2, 64, 32, 16, 8, 4, 8]
```

means:

- `2` active hidden layers,
- hidden sizes: `64`, `32`,
- quantization: layer 1 uses `8-bit`, layer 2 uses `4-bit`,
- the remaining entries are unused because only 2 layers are active.

## Dataset

Default dataset:

- `sklearn.datasets.load_digits()` - a small handwritten-digit classification dataset that is noticeably more challenging than Wine while still training quickly.

Also supported:

- `sklearn.datasets.load_wine()`

## Run

```bash
./venv/bin/python main.py
```

Optional CLI arguments:

```bash
./venv/bin/python main.py --dataset digits --generations 6 --population-size 12 --epochs 20
```

## Notes

- Quantization is implemented as differentiable fake quantization during training.
- The final report also compares the best evolved architecture against binary and ternary weight baselines.
