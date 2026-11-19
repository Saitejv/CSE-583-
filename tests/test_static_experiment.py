"""Fixed-design grouping, paired inference and frozen-path compatibility."""
import math
from pathlib import Path

import numpy as np
import pytest

from benchmarks.corpus.static_experiment import BLOCKS, METHODS, components, design, evaluate, paired_differences
from benchmarks.corpus.registry import SPECS
from jax_unroll.features import extract_features_from_workload
from jax_unroll.model import CostModel, _design
from jax_unroll.static_features import extract_static_features


def test_grouping_has_eight_disjoint_complete_components():
    names = ['matched_gather', 'matched_scatter_add', 'matched_scatter_min']
    folds = components(names)
    assert len(folds)==8
    flattened = [body for members in folds.values() for body in members]
    assert len(flattened)==len(set(flattened))==len(SPECS)+3
    indexed = next(members for key,members in folds.items() if 'indirect_gather' in key)
    assert set(names)<=set(indexed)


def test_extension_is_additive_and_does_not_change_frozen_forecasts():
    workload = SPECS[0].build(1023,1)
    original = extract_features_from_workload(workload)
    model = CostModel.load(Path('models/cost_cpu_recovery_v1.json'))
    before = model.predict_candidates(original)
    extension = extract_static_features(workload)
    assert not (original.keys() & extension.keys())
    assert model.predict_candidates(dict(original,**extension))==before
    again = extract_features_from_workload(workload)
    original.pop('extraction_ms')
    again.pop('extraction_ms')
    assert original==again
    with pytest.raises(ValueError):
        extract_static_features(SPECS[0].build(1023,2))


def test_design_has_no_names_policy_context_or_factor1_effect():
    workload = SPECS[0].build(1023,1)
    row = dict(extract_features_from_workload(workload), **extract_static_features(workload))
    for block, terms in BLOCKS.items():
        assert len(design(row,block))==8+2*len(terms)
        assert np.allclose(design(row,block),0)
        row2 = dict(row, workload='different', family='different', body_hash='different', B=2, R=9)
        assert design(row,block)==design(row2,block)
    row['factor']=8
    assert design(row,'warm_refit')==_design(row,'memory_work')[3]


def test_bootstrap_reduces_related_variants_to_one_body():
    rows = []
    for body, actual, estimate in [('a',1.,0.),('b',1.,1.)]:
        row = {'body':body,'measured_log_ratio':actual,**dict.fromkeys(METHODS,estimate),'zero':0.}
        rows.extend([row]*(20 if body=='a' else 1))
    delta = paired_differences(rows,'zero')['all_static']
    assert delta['mean']==-.5
    assert delta['ci95']==[-1.,0.]


def test_evaluation_predictions_ignore_test_labels():
    # Two bodies in each component, using real static rows and synthetic labels.
    rows=[]
    repeats={}
    folds={}
    for i,spec in enumerate(SPECS[:4]):
        base=extract_features_from_workload(spec.build(1023,1))
        base.update(extract_static_features(spec.build(1023,1)))
        folds.setdefault(str(i//2),[]).append(spec.name)
        for u in (1,2,4,8):
            rows.append(dict(base,workload=spec.name,family=spec.family,variant=spec.primary_variant,
                             factor=u,dtype='float32',backend='cpu',jax_version='0.12.0.1',jaxlib_version='0.12.0.1',
                             log_warm_us=math.log(10)+.01*i*math.log(u),log_cold_ms=math.log(5),
                             cold_median_ms=5.,warm_median_us=10*math.exp(.01*i*math.log(u))))
        repeats[(spec.name,spec.primary_variant,1023)]=np.zeros((3,3))
    model=CostModel.load(Path('models/cost_cpu_recovery_v1.json'))
    _,detail,_=evaluate(rows,repeats,folds,model)
    changed=[dict(r,log_warm_us=float(r['log_warm_us'])+(.4 if r['factor']>1 else 0))
             if r['workload'] in folds['0'] else r for r in rows]
    _,detail2,_=evaluate(changed,repeats,folds,model)
    for a,b in zip(detail,detail2,strict=True):
        if a['fold']=='0':
            assert all(a[m]==b[m] for m in METHODS)


def test_evaluation_rejects_overlapping_or_uncovered_folds():
    spec=SPECS[0]
    base=dict(extract_features_from_workload(spec.build(1023,1)),dtype='float32',workload=spec.name,
              family=spec.family,variant=spec.primary_variant)
    rows=[dict(base,factor=u) for u in (1,2,4,8)]
    for folds in ({'a':[spec.name],'b':[spec.name]}, {'a':['absent']}):
        with pytest.raises(ValueError,match='exactly once'):
            evaluate(rows,{},folds,None)
