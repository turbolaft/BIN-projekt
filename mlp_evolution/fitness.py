from dataclasses import dataclass

import random

import numpy as np
import torch

from mlp_evolution.config import SearchConfig
from mlp_evolution.data import DatasetBundle
from mlp_evolution.genotype import Genotype
from mlp_evolution.model import QuantizedMLP
from mlp_evolution.train import TrainingResult, evaluate_accuracy, train_model


@dataclass(slots=True)
class EvaluatedIndividual:
    genotype: Genotype
    fitness: float
    val_accuracy: float
    test_accuracy: float
    estimated_model_bits: int
    estimated_compute_cost: int
    training_result: TrainingResult


def evaluate_genotype(
    genotype: Genotype,
    dataset: DatasetBundle,
    config: SearchConfig,
    hidden_quantization_override: list[str | int] | None = None,
) -> EvaluatedIndividual:
    _seed_for_genotype(config.random_seed, genotype, hidden_quantization_override)
    model = QuantizedMLP(
        dataset.input_dim,
        dataset.num_classes,
        genotype,
        config,
        hidden_quantization_override=hidden_quantization_override,
    )
    training_result = train_model(
        model=model,
        train_loader=dataset.train_loader,
        val_loader=dataset.val_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
    )
    test_accuracy = evaluate_accuracy(model, dataset.test_loader, config.device)

    model_bits, compute_cost = estimate_efficiency(
        genotype,
        dataset.input_dim,
        dataset.num_classes,
        hidden_quantization_override=hidden_quantization_override,
    )
    fitness = compute_fitness(
        training_result.best_val_accuracy,
        model_bits,
        compute_cost,
        dataset.input_dim,
        dataset.num_classes,
        config,
    )

    return EvaluatedIndividual(
        genotype=genotype,
        fitness=fitness,
        val_accuracy=training_result.best_val_accuracy,
        test_accuracy=test_accuracy,
        estimated_model_bits=model_bits,
        estimated_compute_cost=compute_cost,
        training_result=training_result,
    )


def _seed_for_genotype(
    base_seed: int,
    genotype: Genotype,
    hidden_quantization_override: list[str | int] | None,
) -> None:
    chromosome = tuple(genotype.to_chromosome())
    override = tuple(hidden_quantization_override or [])
    seed = base_seed + abs(hash((chromosome, override))) % 1_000_000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def estimate_efficiency(
    genotype: Genotype,
    input_dim: int,
    num_classes: int,
    hidden_quantization_override: list[str | int] | None = None,
) -> tuple[int, int]:
    dims = [input_dim, *genotype.active_hidden_sizes(), num_classes]
    active_bits = hidden_quantization_override or genotype.active_quantization_bits()
    weight_bits = [*active_bits, 32]

    total_model_bits = 0
    total_compute_cost = 0
    for in_dim, out_dim, bits in zip(dims[:-1], dims[1:], weight_bits, strict=True):
        numeric_bits = _quantization_cost(bits)
        params = (in_dim * out_dim) + out_dim
        total_model_bits += params * numeric_bits
        total_compute_cost += in_dim * out_dim * numeric_bits
    return total_model_bits, total_compute_cost


def _quantization_cost(bits: str | int) -> int:
    if isinstance(bits, int):
        return bits
    if bits == "binary":
        return 1
    if bits == "ternary":
        return 2
    if bits == "fp32":
        return 32
    raise ValueError(f"Unsupported quantization cost for scheme: {bits}")


def compute_fitness(
    val_accuracy: float,
    model_bits: int,
    compute_cost: int,
    input_dim: int,
    num_classes: int,
    config: SearchConfig,
) -> float:
    reference_model_bits = _reference_model_bits(input_dim, num_classes, config)
    reference_compute_cost = _reference_compute_cost(input_dim, num_classes, config)

    model_penalty = model_bits / reference_model_bits
    compute_penalty = compute_cost / reference_compute_cost
    efficiency_penalty = (
        config.model_size_penalty_weight * model_penalty
        + config.compute_penalty_weight * compute_penalty
    )

    return (
        config.accuracy_weight * val_accuracy
        - config.efficiency_weight * efficiency_penalty
    )


def _reference_model_bits(input_dim: int, num_classes: int, config: SearchConfig) -> int:
    max_width = max(config.hidden_size_choices)
    dims = [input_dim] + [max_width] * config.max_hidden_layers + [num_classes]
    bits = [32] * (len(dims) - 1)
    total = 0
    for in_dim, out_dim, bit in zip(dims[:-1], dims[1:], bits, strict=True):
        total += ((in_dim * out_dim) + out_dim) * bit
    return max(total, 1)


def _reference_compute_cost(input_dim: int, num_classes: int, config: SearchConfig) -> int:
    max_width = max(config.hidden_size_choices)
    dims = [input_dim] + [max_width] * config.max_hidden_layers + [num_classes]
    bits = [32] * (len(dims) - 1)
    total = 0
    for in_dim, out_dim, bit in zip(dims[:-1], dims[1:], bits, strict=True):
        total += in_dim * out_dim * bit
    return max(total, 1)
