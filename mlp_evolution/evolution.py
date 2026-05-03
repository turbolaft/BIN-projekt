import argparse
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from mlp_evolution.config import SearchConfig, override_config
from mlp_evolution.data import load_dataset
from mlp_evolution.fitness import EvaluatedIndividual, evaluate_genotype
from mlp_evolution.genotype import Genotype, clone_genotype, crossover, random_genotype


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tournament_selection(
    population: list[EvaluatedIndividual],
    tournament_size: int,
    rng: random.Random,
) -> EvaluatedIndividual:
    contestants = rng.sample(population, k=tournament_size)
    return max(contestants, key=lambda item: item.fitness)


def evolve(config: SearchConfig) -> tuple[EvaluatedIndividual, list[EvaluatedIndividual]]:
    set_seed(config.random_seed)
    rng = random.Random(config.random_seed)
    dataset = load_dataset(config)
    _prepare_log_file(config)

    population = [random_genotype(config, rng) for _ in range(config.population_size)]
    history: list[EvaluatedIndividual] = []
    best_overall: EvaluatedIndividual | None = None
    total_steps = (config.generations * config.population_size) + 3

    with tqdm(
        total=total_steps,
        desc="Program progress",
        disable=not config.verbose,
        leave=True,
        dynamic_ncols=True,
    ) as progress_bar:
        dataset_label = dataset.dataset_metadata.get("name") or f"UCI id={config.dataset_id}"
        _progress_write(
            progress_bar,
            (
                f"Dataset: {dataset_label} | samples="
                f"{dataset.train_size + dataset.val_size + dataset.test_size} | "
                f"input_dim={dataset.input_dim} | classes={dataset.num_classes}"
            ),
        )
        for generation in range(config.generations):
            generation_start = time.perf_counter()
            evaluated: list[EvaluatedIndividual] = []
            for candidate_idx, genotype in enumerate(population, start=1):
                evaluated.append(
                    evaluate_genotype(
                        genotype,
                        dataset,
                        config,
                        hidden_quantization_override=_quantization_override_for_genotype(genotype, config),
                    )
                )
                progress_bar.set_postfix_str(
                    f"generation={generation + 1}/{config.generations}, "
                    f"model={candidate_idx}/{config.population_size}"
                )
                progress_bar.update(1)
            best_candidate_idx = max(
                range(len(evaluated)),
                key=lambda idx: evaluated[idx].fitness,
            )
            evaluated.sort(key=lambda item: item.fitness, reverse=True)

            generation_best = evaluated[0]
            if best_overall is None or generation_best.fitness > best_overall.fitness:
                best_overall = generation_best
            history.extend(evaluated)

            _print_generation_summary(generation, generation_best, evaluated, config, progress_bar)
            _append_generation_log(
                config,
                dataset,
                generation,
                generation_best,
                best_candidate_idx,
                time.perf_counter() - generation_start,
            )

            elites = [clone_genotype(item.genotype) for item in evaluated[: config.elite_count]]
            next_population: list[Genotype] = elites

            while len(next_population) < config.population_size:
                parent_a = tournament_selection(evaluated, config.tournament_size, rng)
                parent_b = tournament_selection(evaluated, config.tournament_size, rng)
                child = crossover(parent_a.genotype, parent_b.genotype, config, rng)
                next_population.append(child)

            population = next_population

        assert best_overall is not None
        compare_extremes(best_overall.genotype, dataset, config, progress_bar)
    return best_overall, history


def compare_extremes(best_genotype: Genotype, dataset, config: SearchConfig, progress_bar: tqdm) -> None:
    best_result = evaluate_genotype(
        best_genotype,
        dataset,
        config,
        hidden_quantization_override=_quantization_override_for_genotype(best_genotype, config),
    )
    progress_bar.set_postfix_str("final=best evolved")
    progress_bar.update(1)
    binary_result = evaluate_genotype(
        best_genotype,
        dataset,
        config,
        hidden_quantization_override=["binary"] * best_genotype.num_hidden_layers,
    )
    progress_bar.set_postfix_str("final=binary")
    progress_bar.update(1)
    ternary_result = evaluate_genotype(
        best_genotype,
        dataset,
        config,
        hidden_quantization_override=["ternary"] * best_genotype.num_hidden_layers,
    )
    progress_bar.set_postfix_str("final=ternary")
    progress_bar.update(1)

    _progress_write(progress_bar, "\nComparison against extreme quantization baselines")
    _progress_write(
        progress_bar,
        f"Best evolved : test_acc={best_result.test_accuracy:.4f}, "
        f"model_bits={best_result.estimated_model_bits}, "
        f"compute_cost={best_result.estimated_compute_cost}",
    )
    _progress_write(
        progress_bar,
        f"Binary       : test_acc={binary_result.test_accuracy:.4f}, "
        f"model_bits={binary_result.estimated_model_bits}, "
        f"compute_cost={binary_result.estimated_compute_cost}",
    )
    _progress_write(
        progress_bar,
        f"Ternary      : test_acc={ternary_result.test_accuracy:.4f}, "
        f"model_bits={ternary_result.estimated_model_bits}, "
        f"compute_cost={ternary_result.estimated_compute_cost}",
    )


def _print_generation_summary(
    generation: int,
    best: EvaluatedIndividual,
    evaluated: list[EvaluatedIndividual],
    config: SearchConfig,
    progress_bar: tqdm,
) -> None:
    if not config.verbose:
        return

    mean_fitness = sum(item.fitness for item in evaluated) / len(evaluated)
    _progress_write(
        progress_bar,
        f"Generation {generation + 1}/{config.generations} | "
        f"best_fitness={best.fitness:.4f} | "
        f"val_acc={best.val_accuracy:.4f} | "
        f"test_acc={best.test_accuracy:.4f} | "
        f"mean_fitness={mean_fitness:.4f}",
    )
    _progress_write(
        progress_bar,
        f"  Best genotype: {best.genotype.to_chromosome()} -> {best.genotype.summary()}",
    )


def _quantization_override_for_genotype(
    genotype: Genotype,
    config: SearchConfig,
) -> list[str | int] | None:
    if config.quant_mode == "mixed":
        return None
    if config.quant_mode == "fp32":
        return ["fp32"] * genotype.num_hidden_layers
    if config.quant_mode == "binary":
        return ["binary"] * genotype.num_hidden_layers
    raise ValueError(f"Unsupported quantization mode: {config.quant_mode}")


def _prepare_log_file(config: SearchConfig) -> None:
    if not config.log_file:
        return
    log_dir = os.path.dirname(config.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(config.log_file, "w", encoding="utf-8"):
        pass


def _append_generation_log(
    config: SearchConfig,
    dataset,
    generation: int,
    best: EvaluatedIndividual,
    best_candidate_idx: int,
    elapsed_seconds: float,
) -> None:
    if not config.log_file:
        return
    line = (
        f"generation: {generation + 1} "
        f"time: {elapsed_seconds:.1f}s "
        f"accuracy: {best.val_accuracy:.4f} "
        f"fitness: {best.fitness:.4f} "
        f"cost_bits: {best.estimated_model_bits} "
        f"genotype: {best.genotype.to_chromosome()}\n"
    )
    with open(config.log_file, "a", encoding="utf-8") as log_handle:
        log_handle.write(line)


def _progress_write(progress_bar: tqdm, message: str) -> None:
    if progress_bar.disable:
        print(message)
        return
    progress_bar.write(message)


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Evolutionary search for a quantized MLP classifier.")
    parser.add_argument("--dataset", dest="dataset_id", type=int)
    parser.add_argument("--seed", dest="random_seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--log-file", dest="log_file", type=str, required=True, help="Path to save the run logs")
    parser.add_argument("--quant-mode", dest="quant_mode", type=str, choices=["fp32", "mixed", "binary"], required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--population-size", dest="population_size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--elite-count", dest="elite_count", type=int)
    parser.add_argument("--mutation-rate", dest="mutation_rate", type=float)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--verbose", action="store_true", default=None)
    parser.add_argument("--quiet", action="store_false", dest="verbose")
    return vars(parser.parse_args())


def main() -> None:
    config = override_config(SearchConfig(), parse_args())
    best, history = evolve(config)
    history.sort(key=lambda item: item.fitness, reverse=True)

    print("\nFinal result")
    print(f"Best genotype: {best.genotype.to_chromosome()}")
    print(f"Architecture : {best.genotype.summary()}")
    print(f"Best fitness : {best.fitness:.4f}")
    print(f"Val accuracy : {best.val_accuracy:.4f}")
    print(f"Test accuracy: {best.test_accuracy:.4f}")
    print(
        f"Efficiency   : model_bits={best.estimated_model_bits}, "
        f"compute_cost={best.estimated_compute_cost}"
    )

    print("\nTop candidates")
    for rank, item in enumerate(history[: config.report_top_k], start=1):
        print(
            f"{rank}. fitness={item.fitness:.4f}, test_acc={item.test_accuracy:.4f}, "
            f"genotype={item.genotype.to_chromosome()}"
        )
