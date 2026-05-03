from dataclasses import dataclass


@dataclass(slots=True)
class SearchConfig:
    dataset_id: int = 222
    random_seed: int = 42
    log_file: str | None = None
    quant_mode: str = "mixed"
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 25
    population_size: int = 12
    generations: int = 6
    elite_count: int = 3
    tournament_size: int = 3
    mutation_rate: float = 0.25
    crossover_rate: float = 0.9
    max_hidden_layers: int = 3
    hidden_size_choices: tuple[int, ...] = (8, 16, 32, 64, 96, 128)
    quantization_bits_choices: tuple[int, ...] = (2, 4, 8, 16, 32)
    binary_compare_bits: int = 1
    ternary_compare_bits: int = 2
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    device: str = "cpu"
    accuracy_weight: float = 0.8
    efficiency_weight: float = 0.2
    model_size_penalty_weight: float = 0.6
    compute_penalty_weight: float = 0.4
    activation_quantization: bool = True
    search_activation_bits_together: bool = True
    verbose: bool = True
    report_top_k: int = 3

    def validate(self) -> None:
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-8:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if self.elite_count >= self.population_size:
            raise ValueError("elite_count must be smaller than population_size")
        if self.max_hidden_layers < 1:
            raise ValueError("max_hidden_layers must be at least 1")
        if self.quant_mode not in {"fp32", "mixed", "binary"}:
            raise ValueError("quant_mode must be one of: 'fp32', 'mixed', 'binary'")


def override_config(config: SearchConfig, args: dict) -> SearchConfig:
    for key, value in args.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    if config.population_size <= 1:
        raise ValueError("population_size must be at least 2")
    config.elite_count = min(config.elite_count, config.population_size - 1)
    config.tournament_size = min(config.tournament_size, config.population_size)
    config.validate()
    return config
