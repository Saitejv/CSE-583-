"""Separate workload-group manifest for the predeclared Phase 3 recovery."""

RECOVERY_SPLIT_VERSION = 1
PINNED_JAX = "0.12.0.1"
TRIPS = (64, 256, 1023)
VARIANTS = ("n64", "n512")
REPEATS = 3
LEGACY_DEV = {
    "scalar_recurrence": "recurrence", "elementwise_sine": "elementwise",
    "reduction_vector": "reduction", "dense_matvec": "dense_linear_algebra",
}
NEW_DEV = {
    "window_read": "window", "window_replace": "window",
    "gather_lookup": "gather", "gather_rows": "gather",
    "scatter_accumulate": "scatter", "scatter_blend": "scatter",
}
CONFIRMATION = {
    "strided_mix": "strided_read", "bucket_max": "histogram",
    "masked_rows": "masked_read", "ring_feedback": "ring_buffer",
}
DEV = LEGACY_DEV | NEW_DEV
PARTITIONS = {"dev": NEW_DEV, "confirmation": CONFIRMATION}


def check_rows(rows, partition):
    allowed = {"dev": DEV, "confirmation": CONFIRMATION}[partition]
    if not rows or any(
        row["workload"] not in allowed or allowed[row["workload"]] != row["family"] or row["split"] != partition
        or (row["backend"], row["jax_version"], row["jaxlib_version"]) != ("cpu", PINNED_JAX, PINNED_JAX)
        for row in rows
    ):
        raise ValueError(f"recovery requires pinned CPU {partition} rows from the frozen workload manifest")


def manifest() -> dict:
    return {
        "recovery_split_version": RECOVERY_SPLIT_VERSION,
        "legacy_dev": LEGACY_DEV, "new_dev": NEW_DEV, "confirmation": CONFIRMATION,
        "trip_counts": TRIPS, "variants": VARIANTS, "process_repeats": REPEATS,
    }
