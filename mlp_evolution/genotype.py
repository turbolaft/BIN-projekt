from dataclasses import dataclass
import random

from mlp_evolution.config import SearchConfig


@dataclass(slots=True)
class Genotype:
    num_hidden_layers: int
    hidden_sizes: list[int]
    quantization_bits: list[int]

    def active_hidden_sizes(self) -> list[int]:
        return self.hidden_sizes[: self.num_hidden_layers]

    def active_quantization_bits(self) -> list[int]:
        return self.quantization_bits[: self.num_hidden_layers]

    def to_chromosome(self) -> list[int]:
        return [self.num_hidden_layers, *self.hidden_sizes, *self.quantization_bits]

    def summary(self) -> str:
        layers = self.active_hidden_sizes()
        bits = self.active_quantization_bits()
        pairs = [f"{size}@{bit}b" for size, bit in zip(layers, bits, strict=True)]
        return f"layers={self.num_hidden_layers}, architecture=[{', '.join(pairs)}]"


def random_genotype(config: SearchConfig, rng: random.Random) -> Genotype:
    hidden_sizes = [rng.choice(config.hidden_size_choices) for _ in range(config.max_hidden_layers)]
    quantization_bits = _initial_quantization_bits(config, rng)
    return Genotype(
        num_hidden_layers=rng.randint(1, config.max_hidden_layers),
        hidden_sizes=hidden_sizes,
        quantization_bits=quantization_bits,
    )


def crossover(parent_a: Genotype, parent_b: Genotype, config: SearchConfig, rng: random.Random) -> Genotype:
    if rng.random() > config.crossover_rate:
        child = clone_genotype(rng.choice([parent_a, parent_b]))
        repair(child, config, rng)
        return child

    child_hidden_sizes = []
    for a_size, b_size in zip(parent_a.hidden_sizes, parent_b.hidden_sizes, strict=True):
        child_hidden_sizes.append(rng.choice([a_size, b_size]))
    child_bits = _crossover_quantization_bits(parent_a, parent_b, config, rng)

    child_layers = rng.choice([parent_a.num_hidden_layers, parent_b.num_hidden_layers])
    child = Genotype(child_layers, child_hidden_sizes, child_bits)
    mutate(child, config, rng)
    repair(child, config, rng)
    return child


def mutate(genotype: Genotype, config: SearchConfig, rng: random.Random) -> None:
    if rng.random() < config.mutation_rate:
        genotype.num_hidden_layers = rng.randint(1, config.max_hidden_layers)

    for idx in range(config.max_hidden_layers):
        if rng.random() < config.mutation_rate:
            genotype.hidden_sizes[idx] = rng.choice(config.hidden_size_choices)
        if config.quant_mode == "mixed" and rng.random() < config.mutation_rate:
            genotype.quantization_bits[idx] = rng.choice(config.quantization_bits_choices)


def repair(genotype: Genotype, config: SearchConfig, rng: random.Random) -> None:
    genotype.num_hidden_layers = max(1, min(config.max_hidden_layers, genotype.num_hidden_layers))

    while len(genotype.hidden_sizes) < config.max_hidden_layers:
        genotype.hidden_sizes.append(rng.choice(config.hidden_size_choices))
    while len(genotype.quantization_bits) < config.max_hidden_layers:
        genotype.quantization_bits.append(_default_quantization_bit(config))

    genotype.hidden_sizes = [
        size if size in config.hidden_size_choices else rng.choice(config.hidden_size_choices)
        for size in genotype.hidden_sizes[: config.max_hidden_layers]
    ]
    genotype.quantization_bits = _repair_quantization_bits(genotype.quantization_bits, config, rng)


def clone_genotype(genotype: Genotype) -> Genotype:
    return Genotype(
        num_hidden_layers=genotype.num_hidden_layers,
        hidden_sizes=list(genotype.hidden_sizes),
        quantization_bits=list(genotype.quantization_bits),
    )


def _initial_quantization_bits(config: SearchConfig, rng: random.Random) -> list[int]:
    fixed_bits = _fixed_quantization_bits(config)
    if fixed_bits is not None:
        return fixed_bits
    return [rng.choice(config.quantization_bits_choices) for _ in range(config.max_hidden_layers)]


def _crossover_quantization_bits(
    parent_a: Genotype,
    parent_b: Genotype,
    config: SearchConfig,
    rng: random.Random,
) -> list[int]:
    fixed_bits = _fixed_quantization_bits(config)
    if fixed_bits is not None:
        return fixed_bits
    child_bits = []
    for a_bit, b_bit in zip(parent_a.quantization_bits, parent_b.quantization_bits, strict=True):
        child_bits.append(rng.choice([a_bit, b_bit]))
    return child_bits


def _repair_quantization_bits(bits: list[int], config: SearchConfig, rng: random.Random) -> list[int]:
    fixed_bits = _fixed_quantization_bits(config)
    if fixed_bits is not None:
        return fixed_bits
    return [
        bit if bit in config.quantization_bits_choices else rng.choice(config.quantization_bits_choices)
        for bit in bits[: config.max_hidden_layers]
    ]


def _default_quantization_bit(config: SearchConfig) -> int:
    fixed_bits = _fixed_quantization_bits(config)
    if fixed_bits is not None:
        return fixed_bits[0]
    return config.quantization_bits_choices[0]


def _fixed_quantization_bits(config: SearchConfig) -> list[int] | None:
    if config.quant_mode == "fp32":
        return [32] * config.max_hidden_layers
    if config.quant_mode == "binary":
        return [config.binary_compare_bits] * config.max_hidden_layers
    return None
