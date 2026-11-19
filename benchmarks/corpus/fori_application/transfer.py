"""Prospective real-data transfer of the locked matrix policy."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from benchmarks.corpus.fori_application.protocol import OUT, HORIZONS, ARMS
from benchmarks.corpus.fori_regret.acquire import REPEATS, measure, run_workers
from benchmarks.corpus.fori_regret.workloads import Case


def weighted_covariance(size, trip, factor):
    """Compute a class-balanced feature covariance by weighted online centering."""
    from sklearn.datasets import load_breast_cancer
    x, y = load_breast_cancer(return_X_y=True)
    x = ((x-x.mean(axis=0)) / x.std(axis=0)).astype(np.float32)
    weights = (len(x) / (2*np.bincount(y)[y])).astype(np.float32)
    assert trip == len(x)

    def run(x, weights, mean, scatter):
        def step(i, state):
            mass, mean, scatter = state
            sample, weight = x[i], weights[i]
            new_mass = mass + weight
            delta = sample - mean
            new_mean = mean + (weight/new_mass)*delta
            return new_mass, new_mean, scatter + weight * delta[:,None] * (sample-new_mean)[None,:]
        mass, mean, scatter = lax.fori_loop(0, trip, step, (np.float32(0), mean, scatter), unroll=factor)
        return mean, scatter/mass

    return run, (x, weights, np.zeros(x.shape[1],np.float32), np.zeros((x.shape[1],x.shape[1]),np.float32))


def signal_grid(size, trip, factor):
    """Replay lagged entry/exit signals and turnover costs on historical daily returns."""
    from matplotlib.cbook import get_sample_data
    with np.load(get_sample_data("goog.npz", asfileobj=False)) as archive:
        close = archive["price_data"]["adj_close"]
    log_price = np.log(close)
    # The signal is known at the close before the return being accumulated.
    signal = (log_price[10:-1] - log_price[:-11])[-trip:].astype(np.float32)
    returns = np.diff(log_price)[10:][-trip:].astype(np.float32)
    entry = np.linspace(.002,.12,32,dtype=np.float32)
    exit_level = np.linspace(-.12,-.002,32,dtype=np.float32)
    assert signal.size == returns.size == trip

    def run(signal, returns, enter, leave, initial, pnl):
        def step(i, state):
            held, net = state
            momentum = signal[i]
            buy = momentum > enter[:,None]
            keep = momentum >= leave[None,:]
            position = buy | (held & keep)
            return position, net + position*returns[i] - np.float32(.001)*(held != position)
        return lax.fori_loop(0, trip, step, (initial, pnl), unroll=factor)

    return run, (signal, returns, entry, exit_level, np.zeros((32,32),bool),np.zeros((32,32),np.float32))


def terrain_interpolation(size, trip, factor):
    """Interpolate surveyed elevations by nearest sample on a regular evaluation grid."""
    from matplotlib.cbook import get_sample_data
    with np.load(get_sample_data("jacksboro_fault_dem.npz", asfileobj=False)) as archive:
        dem = archive["elevation"]
        dx, dy = abs(float(archive["dx"])), abs(float(archive["dy"]))
    cells = np.random.default_rng(5831925).choice(np.flatnonzero(np.isfinite(dem)),trip,replace=False)
    iy, ix = np.unravel_index(cells, dem.shape)
    x = np.linspace(0, (dem.shape[1]-1)*dx, 48,dtype=np.float32)
    y = np.linspace(0, (dem.shape[0]-1)*dy, 40,dtype=np.float32)
    shape = (y.size,x.size)

    def run(sx, sy, heights, x, y, distance, index):
        def step(i, state):
            best, nearest = state
            d = (x-sx[i])[None,:]**2 + (y-sy[i])[:,None]**2
            return jnp.minimum(best,d), jnp.where(d<best,i,nearest)
        _, nearest = lax.fori_loop(0,trip,step,(distance,index),unroll=factor)
        return heights[nearest]

    return run, ((ix*dx).astype(np.float32),(iy*dy).astype(np.float32),dem[iy,ix].astype(np.float32),
                 x,y,np.full(shape,np.inf,np.float32), np.zeros(shape,np.int32))


def recursive_least_squares(size, trip, factor):
    """Fit coefficients and inverse normal matrix by recursive ridge regression."""
    from sklearn.datasets import load_diabetes
    x,y = load_diabetes(return_X_y=True)
    x = np.column_stack([np.ones(len(x)),x*np.sqrt(len(x))]).astype(np.float32)
    y = ((y-y.mean())/y.std()).astype(np.float32)
    assert len(y) == trip

    def run(x,y,beta,inv):
        def step(i, state):
            coef, cov = state
            row = x[i]
            projection = cov @ row
            gain = projection / (1 + row @ projection)
            return coef + gain*(y[i]-row@coef), cov - gain[:,None]*projection[None,:]
        return lax.fori_loop(0,trip,step,(beta,inv),unroll=factor)

    return run, (x,y,np.zeros(x.shape[1],np.float32),np.eye(x.shape[1],dtype=np.float32))


def settings():
    return [(Case(fn.__name__, fn.__name__, "transfer", fn), "real", trip)
            for fn,trip in ((weighted_covariance,569), (signal_grid,1024),
                            (terrain_interpolation,512), (recursive_least_squares,442))]


def check(create=False):
    inputs = {s.name:[sha256(np.asarray(x).tobytes()).hexdigest() for x in jax.tree.leaves(s.build(t,1,v).args)]
              for s,v,t in settings()}
    content = dict(source=sha256(Path(__file__).read_bytes()).hexdigest(), inputs=inputs,
                   policy=sha256((OUT/"fit.json").read_bytes()).hexdigest(), arms=ARMS,
                   calls=HORIZONS, repeats=REPEATS, scope="three streamed applications plus a dot-product control")
    path = OUT / "transfer_manifest.json"
    encoded = json.dumps(content,indent=2)+"\n"
    if create and not path.exists():
        with path.open("x") as file:
            file.write(encoded)
    if encoded != path.read_text():
        raise ValueError("transfer cases or calibrated policy changed")


if __name__ == "__main__":
    stage,*extra = sys.argv[1:]
    check(create=stage=="freeze")
    if extra:
        print(json.dumps(measure(settings(),int(extra[1]))))
    elif stage=="sweep":
        run_workers(__spec__.name,["sweep"],REPEATS,OUT/"transfer_runs.json")
