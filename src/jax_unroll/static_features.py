"""Experimental v2 extension; never changes the frozen v1 feature dictionary."""
from jax_unroll.features import extract_loop_body
from jax_unroll.static_structure import structure_features
from jax_unroll.static_indices import index_features

STATIC_FEATURE_VERSION = 2
STRUCTURE_TERMS = ('carry_slice_count', 'carry_concatenate_count', 'carry_slice_bytes',
                   'carry_updated_bytes', 'carry_dependency_pairs', 'carry_to_carry_max_depth')
INDEX_TERMS = ('index_gather_count', 'index_scatter_count', 'index_scatter_add_count',
               'index_scatter_min_count', 'index_scatter_other_count', 'index_from_constant',
               'index_from_counter', 'index_from_scanned', 'index_from_carry',
               'index_from_unknown', 'index_mixed_sources_count')


def extract_static_features(workload):
    """Trace only factor 1; values and candidate compiler IR are never inspected."""
    if workload.factor != 1:
        raise ValueError('experimental extension requires a factor-1 workload')
    loop = extract_loop_body(workload.fn, workload.args, loop_kind=workload.loop_kind)
    return {'static_feature_version': STATIC_FEATURE_VERSION,
            **structure_features(loop), **index_features(loop)}
