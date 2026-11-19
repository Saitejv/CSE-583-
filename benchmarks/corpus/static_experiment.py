"""Fixed-design static-feature development experiment; no gates or model writes."""
import argparse
from collections import defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from benchmarks.corpus import pilot
from benchmarks.corpus.registry import SPECS, build_workload, related_groups
from benchmarks.dataset import aggregate_labels, build_feature_table
from benchmarks.oracle import load_raw
from benchmarks.recovery_data import JOIN_KEY, MODEL, VERSIONS, config_key, dump, require_cpu, sha, write_csv
from benchmarks.recovery_metrics import paired_rows
from jax_unroll.features import extract_features_from_workload
from jax_unroll.model import CostModel, LogCosts, _design, group_configs
from jax_unroll.schema import FACTORS
from jax_unroll.static_features import INDEX_TERMS, STRUCTURE_TERMS, extract_static_features
from jax_unroll.validation import forecast

OUT = Path('results/static_features')
PROTOCOL = Path('docs/static_feature_protocol.md')
METHODS = ('zero', 'factor_median', 'frozen', 'warm_refit', 'structure', 'indices', 'all_static')
BLOCKS = {'warm_refit': (), 'structure': STRUCTURE_TERMS, 'indices': INDEX_TERMS,
          'all_static': STRUCTURE_TERMS + INDEX_TERMS}


def design(row, block):
    u = int(row['factor'])
    extra = tuple(np.log1p(float(row[key])) / 8 for key in BLOCKS[block])
    return (*_design(row, 'memory_work')[3],
            *(basis * value for basis in (1 - 1/u, np.log(u)) for value in extra))


def components(matched_names=()):
    folds = pilot.family_components(SPECS, related_groups())
    indexed = next(key for key in folds if 'indirect_gather' in key)
    folds[indexed] = sorted([*folds[indexed], *matched_names])
    return folds


def metrics(rows):
    return {method: {'mae': float(np.mean([abs(row[method]-row['measured_log_ratio']) for row in rows])),
                     'bias': float(np.mean([row[method]-row['measured_log_ratio'] for row in rows]))}
            for method in METHODS}


def paired_differences(rows, baseline):
    errors = defaultdict(list)
    for row in rows:
        actual = row['measured_log_ratio']
        errors[row['body']].append([abs(row[m]-actual)-abs(row[baseline]-actual) for m in METHODS])
    values = np.array([np.mean(errors[b], axis=0) for b in sorted(errors)])
    draws = np.random.default_rng(583).integers(len(values), size=(2000, len(values)))
    lo, hi = np.quantile(np.mean(values[draws], axis=1), [.025, .975], axis=0)
    return {m: {'mean': float(mean), 'ci95': [float(l), float(h)]}
            for m, mean, l, h in zip(METHODS, values.mean(axis=0), lo, hi, strict=True)}


def evaluate(rows, repeats, folds, model):
    groups = {config_key(g[1]): g for g in group_configs(rows)}
    observed_bodies = {key[0] for key in groups}
    memberships = [name for members in folds.values() for name in members if name in observed_bodies]
    if len(memberships) != len(set(memberships)) or set(memberships) != observed_bodies:
        raise ValueError('folds must cover each observed body exactly once')
    cases = {config_key(c.features): c for c in forecast(model, rows)}
    detailed, fold_reports = [], {}
    predictions = {m: [] for m in METHODS}
    for fold, members in folds.items():
        test = sorted(k for k in groups if k[0] in members)
        train = sorted(set(groups)-set(test))
        if not test or not train:
            raise ValueError('empty fold')
        if {k[0] for k in train} & {k[0] for k in test}:
            raise ValueError('body leakage')
        def ratio(k, u):
            return float(groups[k][u]['log_warm_us'])-float(groups[k][1]['log_warm_us'])
        fits = {block: Ridge(alpha=.025, fit_intercept=False).fit(
            [design(groups[k][u], block) for k in train for u in FACTORS],
            [ratio(k, u) for k in train for u in FACTORS]) for block in BLOCKS}
        medians = {u: float(np.median([ratio(k, u) for k in train])) for u in FACTORS}
        fold_rows = []
        for key in test:
            case = cases[key]
            base = case.predictions[1].log_warm_us
            effects = {m: {1: 0.} for m in METHODS}
            for j, u in enumerate(FACTORS[1:]):
                effect = {m: float(fits[m].predict([design(groups[key][u], m)])[0]) for m in BLOCKS}
                effect.update(zero=0., factor_median=medians[u], frozen=case.predictions[u].log_warm_us-base)
                r = repeats[key][:, j]
                fold_rows.append({'body': key[0], 'variant': key[1], 'trip_count': key[2], 'factor': u,
                                  'fold': fold, 'measured_log_ratio': ratio(key, u),
                                  'repeat_sign_consistent': bool(np.all(np.sign(r)==np.sign(r[0]))),
                                  'repeat_log_range': float(np.ptp(r)), **effect})
                for m in METHODS:
                    effects[m][u] = effect[m]
            for m in METHODS:
                pred = {u: LogCosts(case.predictions[u].log_cold_ms, base+effects[m][u]) for u in FACTORS}
                predictions[m].append(replace(case, predictions=pred))
        fold_reports[fold] = {'train_configs': train, 'test_configs': test, 'metrics': metrics(fold_rows),
                             'paired_vs_zero': paired_differences(fold_rows, 'zero')}
        detailed.extend(fold_rows)
    pooled = metrics(detailed)
    delta = {b: paired_differences(detailed, b) for b in ('zero', 'factor_median', 'warm_refit')}
    wins = sum(f['metrics']['all_static']['mae'] < f['metrics']['zero']['mae'] for f in fold_reports.values())
    promising = (pooled['all_static']['mae'] < min(pooled[b]['mae'] for b in ('zero','factor_median'))
                 and wins > len(folds)/2 and delta['zero']['all_static']['ci95'][1] < 0)
    report = {'warm_metrics': pooled, 'paired_mae_differences': delta, 'folds': fold_reports,
              'primary_folds_beating_zero': wins, 'primary_promising': promising,
              'repeat_stability': {'stable': sum(r['repeat_sign_consistent'] for r in detailed),
                                   'comparisons': len(detailed),
                                   'median_log_range': float(np.median([r['repeat_log_range'] for r in detailed]))},
              'diagnostic_replay_without_decision_overhead': {m: pilot.policy_metrics(c) for m,c in predictions.items()}}
    return report, detailed, predictions


def prepare(raw, builder):
    labels = aggregate_labels(raw)
    features = build_feature_table(labels, workload_builder=builder).drop(columns='extraction_ms')
    joined = features.merge(labels, on=JOIN_KEY, validate='one_to_one')
    extensions = {}
    repeats = {}
    for key, group in raw.groupby(['workload','variant','trip_count']):
        extensions[key] = extract_static_features(builder(key[0], int(key[2]), 1, key[1]))
        warm = group.pivot(index='process_repeat', columns='factor', values='warm_median_us')
        repeats[key] = np.log(warm[list(FACTORS[1:])].to_numpy()/warm[[1]].to_numpy())
    rows = [dict(r, **VERSIONS, **extensions[config_key(r)]) for r in joined.to_dict('records')]
    return rows, repeats


def matched_validation(raw):
    from benchmarks.corpus.static_matched import SPECS as MATCHED
    expected = {(s.name,v,t) for s in MATCHED for v in s.shape_variants for t in s.trip_counts}
    observed = set(raw.groupby(['workload','variant','trip_count']).groups)
    if observed - expected:
        raise ValueError('unexpected matched configuration')
    good, dropped = [], {}
    for spec in MATCHED:
        body = raw[raw.workload.eq(spec.name)]
        reason = None
        for key in sorted(k for k in expected if k[0]==spec.name):
            g = body[body.variant.eq(key[1]) & body.trip_count.eq(key[2])]
            if (len(g)!=12 or set(zip(g.factor,g.process_repeat)) != set(product(FACTORS,range(3)))):
                reason = 'incomplete factor/repeat matrix'
            elif (not g.correct.astype(str).str.lower().eq('true').all() or g.error.fillna('').ne('').any()
                  or not np.isfinite(g[['cold_total_ms','warm_median_us']]).all().all()
                  or not (g[['cold_total_ms','warm_median_us']]>0).all().all()):
                reason = 'invalid measurement'
            elif any(not g[c].eq(v).all() for c,v in {'backend':'cpu','device':'cpu:0',**VERSIONS}.items()):
                raise ValueError('unpinned runtime')
            elif (not g.family.eq(spec.family).all() or not g.loop_kind.eq(spec.loop_kind).all()
                  or g['shape'].nunique()!=1 or g.dtype.nunique()!=1):
                raise ValueError('inconsistent matched abstract metadata')
            else:
                expected_workload = spec.build(int(key[2]), 1, key[1])
                if not g['shape'].eq(expected_workload.shape).all() or not g.dtype.eq(expected_workload.dtype).all():
                    raise ValueError('matched metadata differs from declared abstract inputs')
        if reason:
            dropped[spec.name] = reason
        else:
            good.append(body)
    return (pd.concat(good,ignore_index=True) if good else raw.iloc[:0]), dropped


def pair_checks():
    import jax
    from benchmarks.corpus.static_matched import SPECS as MATCHED
    checks = []
    for s in MATCHED:
        variants = list(s.shape_variants)
        for trip in s.trip_counts:
            for u in FACTORS:
                a,b = [s.build(trip,u,v) for v in variants]
                ja,jb = [str(jax.make_jaxpr(w.fn)(*w.args)) for w in (a,b)]
                fa,fb = [extract_features_from_workload(w) for w in (a,b)]
                for f in (fa,fb):
                    f.pop('extraction_ms')
                ea,eb = [extract_static_features(s.build(trip,1,v)) for v in variants]
                if ja!=jb or fa!=fb or ea!=eb:
                    raise ValueError('runtime pair differs in static features or Jaxpr')
                checks.append({'body':s.name,'trip_count':trip,'factor':u,'identical_jaxpr':True,'identical_features':True})
    return checks


def matched_effects(raw):
    result = []
    for (name,trip),g in raw.groupby(['workload','trip_count']):
        variants = sorted(g.variant.unique())
        warm = g.pivot(index=['variant','process_repeat'],columns='factor',values='warm_median_us')
        for u in FACTORS[1:]:
            ratios = np.log(warm[u]/warm[1])
            a,b = [ratios.loc[v].sort_index().to_numpy() for v in variants]
            delta = b-a
            result.append({'body':name,'trip_count':int(trip),'factor':u,'variant_a':variants[0],
                           'variant_b':variants[1], 'a_repeat_log_ratios':a.tolist(),'b_repeat_log_ratios':b.tolist(),
                           'b_minus_a':delta.tolist(),'median_delta':float(np.median(delta)),
                           'delta_sign_consistent':bool(np.all(np.sign(delta)==np.sign(delta[0])))})
    return result


def run(args):
    require_cpu()
    receipt = (OUT/'premeasurement.sha256').read_text().splitlines()
    paths = [PROTOCOL, Path('benchmarks/corpus/manifest.json'), MODEL, pilot.PILOT/'primary_1023_raw.csv']
    if receipt != [f'{sha(p)}  {p}' for p in paths]:
        raise ValueError('premeasurement inputs changed')
    checks = pair_checks()
    dump(OUT/'identical_pairs.json', checks)
    if args.check_pairs:
        print(f'Checked {len(checks)} identical Jaxpr/feature pairs; no timings')
        return
    raw, dropped, _ = pilot.validate_acquisition(load_raw(pilot.PILOT/'primary_1023_raw.csv'))
    rows,repeats = prepare(raw, build_workload)
    matrices = {'pilot':(rows,repeats,components())}
    matched_path = OUT/'matched_raw.csv'
    if matched_path.exists():
        from benchmarks.corpus.static_matched import build_workload as matched_builder
        mraw, mdropped = matched_validation(load_raw(matched_path))
        if len(mraw):
            mrows,mrepeats = prepare(mraw,matched_builder)
            matrices['augmented'] = (rows+mrows,{**repeats,**mrepeats},components(sorted(mraw.workload.unique())))
        dump(OUT/'matched_diagnostics.json',{'dropped':mdropped,'raw_sha256':sha(matched_path),
                                           'effects':matched_effects(mraw)})
    else:
        raise ValueError('matched raw matrix missing; run --check-pairs before acquisition')
    model = CostModel.load(MODEL)
    reports = {}
    for name,(r,rep,folds) in matrices.items():
        report,detail,pred = evaluate(r,rep,folds,model)
        reports[name] = report
        write_csv(OUT/f'{name}_features.csv',r,r[0])
        write_csv(OUT/f'{name}_warm_ratios.csv',detail,detail[0])
        replay = [dict(method=m,**row) for m,cases in pred.items() for row in paired_rows(cases)]
        write_csv(OUT/f'{name}_policy_pairs.csv',replay,replay[0])
        print(name, {m:round(v['mae'],4) for m,v in report['warm_metrics'].items()},'promising',report['primary_promising'])
    dump(OUT/'report.json',{'protocol_sha256':sha(PROTOCOL),'frozen_model_sha256':sha(MODEL),
                          'pilot_raw_sha256':sha(pilot.PILOT/'primary_1023_raw.csv'),
                          'pilot_dropped':dropped,'feature_version':2,'matrices':reports})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-pairs',action='store_true')
    run(p.parse_args())

if __name__=='__main__':
    main()
