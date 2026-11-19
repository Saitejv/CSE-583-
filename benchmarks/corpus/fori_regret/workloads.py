"""Reviewed body families and their pre-measurement split."""

from dataclasses import dataclass
from typing import Callable, NamedTuple

from benchmarks.corpus.fori_regret import agy_cases, opencode_cases


class Workload(NamedTuple):
    fn: Callable
    args: tuple


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    partition: str
    factory: Callable
    atol: float = 1e-4
    rtol: float = 1e-5

    def build(self, trip, factor, variant):
        return Workload(*self.factory(variant, trip, factor))


CASES = (
    Case("negative_binomial_bank", "likelihood_feedback", "train", agy_cases.make_nb_streaming_loglik),
    Case("packet_reorder_scoreboard", "associative_state_writes", "train", agy_cases.packet_reorder_scoreboard_factory),
    Case("spectral_equalizer", "nonlinear_spectral_gain", "holdout", agy_cases.piecewise_spectral_equalizer_factory),
    Case("packed_prefix_rank", "packed_word_rank", "holdout", agy_cases.packed_prefix_rank_factory),
    Case("compressed_trie_walk", "failure_link_traversal", "holdout", agy_cases.compressed_trie_walk_factory),
    Case("branch_tag_cache", "associative_state_writes", "train", opencode_cases.branch_tags),
    Case("discounted_path_softmin", "discounted_cost_reduction", "train", opencode_cases.discounted_softmin),
)


def cases(split):
    return {case.name: case for case in CASES if case.partition == split}
