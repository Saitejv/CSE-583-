"""Numerical simulation independent fori_loop workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats
from benchmarks.corpus.spec import CorpusSpec, Provenance

# 1. sim_verlet_symplectic: Velocity Verlet / Leapfrog Symplectic Integrator [fresh_eval]
def _verlet_factory(variant: str, trip: int, factor: int):
    n_particles = {"n16": 16, "n32": 32, "n64": 64}[variant]
    rng = np.random.default_rng(583_851)
    pos_init = floats(rng, n_particles, 3, scale=2.0)
    vel_init = floats(rng, n_particles, 3, scale=0.1)
    force_init = floats(rng, n_particles, 3, scale=0.1)
    dt = jnp.float32(0.005)

    def fn(pos: jax.Array, vel: jax.Array, force: jax.Array, step_dt: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            p, v, f = carry
            # Verlet step: v(t + dt/2) = v(t) + 0.5 * dt * f(t)
            v_half = v + 0.5 * step_dt * f
            # p(t + dt) = p(t) + dt * v(t + dt/2)
            p_next = p + step_dt * v_half
            # Lennard-Jones style pairwise repulsive force
            # Pairwise displacement: diff[j, k] = p[j] - p[k]
            diff = p_next[:, None, :] - p_next[None, :, :]
            dist_sq = jnp.sum(diff ** 2, axis=-1) + jnp.float32(1e-2)
            inv_dist6 = 1.0 / (dist_sq ** 3)
            f_mag = inv_dist6 * (inv_dist6 - 0.5)
            f_next = jnp.sum(f_mag[:, :, None] * diff, axis=1) * 0.01
            # v(t + dt) = v(t + dt/2) + 0.5 * dt * f(t + dt)
            v_next = v_half + 0.5 * step_dt * f_next
            return (p_next, v_next, f_next)

        return lax.fori_loop(0, trip, body, (pos, vel, force), unroll=factor)

    return built(fn, pos_init, vel_init, force_init, dt)


# 2. sim_reaction_diffusion_1d: Gray-Scott Coupled Reaction-Diffusion PDE Ring
def _reaction_diff_factory(variant: str, trip: int, factor: int):
    grid_size = {"g64": 64, "g128": 128, "g256": 256}[variant]
    rng = np.random.default_rng(583_852)
    init_u = jnp.ones((grid_size,), dtype=jnp.float32)
    init_v = jnp.zeros((grid_size,), dtype=jnp.float32).at[grid_size // 4: 3 * grid_size // 4].set(0.5)
    Du = jnp.float32(0.16)
    Dv = jnp.float32(0.08)
    F = jnp.float32(0.035)
    k = jnp.float32(0.065)
    dt = jnp.float32(1.0)

    def fn(u_init: jax.Array, v_init: jax.Array, d_u: jax.Array, d_v: jax.Array, feed: jax.Array, kill: jax.Array, step_dt: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            u, v = carry
            # 1D periodic Laplacian: roll(x, 1) + roll(x, -1) - 2 * x
            lap_u = jnp.roll(u, 1) + jnp.roll(u, -1) - 2.0 * u
            lap_v = jnp.roll(v, 1) + jnp.roll(v, -1) - 2.0 * v
            uv2 = u * v * v
            du = d_u * lap_u - uv2 + feed * (1.0 - u)
            dv = d_v * lap_v + uv2 - (feed + kill) * v
            return (jnp.clip(u + step_dt * du, 0.0, 1.0), jnp.clip(v + step_dt * dv, 0.0, 1.0))

        return lax.fori_loop(0, trip, body, (u_init, v_init), unroll=factor)

    return built(fn, init_u, init_v, Du, Dv, F, k, dt)


# 3. sim_finite_volume_burgers: Godunov Upwind Non-Linear Burgers' Shock Capturing Solver
def _burgers_factory(variant: str, trip: int, factor: int):
    n = {"n64": 64, "n128": 128, "n256": 256}[variant]
    rng = np.random.default_rng(583_853)
    init_u = jnp.sin(jnp.linspace(0, 2 * np.pi, n, endpoint=False)).astype(jnp.float32)
    dx = jnp.float32(1.0 / n)
    dt = jnp.float32(0.2 * dx)

    def fn(u_init: jax.Array, step_dx: jax.Array, step_dt: jax.Array):
        def body(i: int, carry: jax.Array):
            u = carry
            u_l = u
            u_r = jnp.roll(u, -1)
            # Godunov numerical flux for Burgers' flux f(u) = 0.5 * u^2
            f_l = 0.5 * u_l * u_l
            f_r = 0.5 * u_r * u_r
            # Shock or rarefaction
            flux = jnp.where(u_l >= u_r, jnp.maximum(f_l, f_r),
                             jnp.where(u_l <= 0.0, jnp.where(u_r >= 0.0, 0.0, f_r), f_l))
            flux_in = jnp.roll(flux, 1)
            flux_out = flux
            new_u = u - (step_dt / step_dx) * (flux_out - flux_in)
            return new_u

        return lax.fori_loop(0, trip, body, u_init, unroll=factor)

    return built(fn, init_u, dx, dt)


# 4. sim_pbd_cloth_spring: Position-Based Dynamics Distance Constraint Projection
def _pbd_factory(variant: str, trip: int, factor: int):
    n = {"n32": 32, "n64": 64, "n128": 128}[variant]
    rng = np.random.default_rng(583_854)
    pos_init = floats(rng, n, 3, scale=1.0)
    prev_init = pos_init.copy()
    rest_len = jnp.float32(0.1)
    stiffness = jnp.float32(0.5)

    def fn(pos: jax.Array, prev_pos: jax.Array, r_len: jax.Array, k_stiff: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            p, p_prev = carry
            # Verlet position prediction
            v = p - p_prev
            p_pred = p + v + jnp.array([0.0, -0.001, 0.0], dtype=jnp.float32)
            # Distance constraint projection across chain
            p_next = jnp.roll(p_pred, -1, axis=0)
            diff = p_pred - p_next
            dist = jnp.linalg.norm(diff, axis=-1, keepdims=True) + 1e-6
            corr = 0.5 * k_stiff * (dist - r_len) * (diff / dist)
            p_corrected = p_pred - corr + jnp.roll(corr, 1, axis=0)
            # Anchor first particle
            p_corrected = p_corrected.at[0].set(pos[0])
            return (p_corrected, p)

        return lax.fori_loop(0, trip, body, (pos, prev_pos), unroll=factor)

    return built(fn, pos_init, prev_init, rest_len, stiffness)


# 5. sim_shallow_water_1d: 1D Saint-Venant Shallow Water Equations with Lax-Friedrichs Flux [fresh_eval]
def _shallow_water_factory(variant: str, trip: int, factor: int):
    grid = {"g64": 64, "g128": 128, "g256": 256}[variant]
    rng = np.random.default_rng(583_855)
    init_h = jnp.ones((grid,), dtype=jnp.float32).at[grid // 3: 2 * grid // 3].add(0.5)
    init_hu = jnp.zeros((grid,), dtype=jnp.float32)
    g = jnp.float32(9.81)
    dx = jnp.float32(1.0 / grid)
    dt = jnp.float32(0.001)

    def fn(h_init: jax.Array, hu_init: jax.Array, grav: jax.Array, step_dx: jax.Array, step_dt: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            h, hu = carry
            u = hu / (h + 1e-5)
            # Fluxes: F_h = hu, F_hu = hu * u + 0.5 * g * h^2
            f_h = hu
            f_hu = hu * u + 0.5 * grav * (h ** 2)
            # Lax-Friedrichs interface numerical flux
            f_h_r = 0.5 * (f_h + jnp.roll(f_h, -1)) - 0.5 * (step_dx / step_dt) * (jnp.roll(h, -1) - h)
            f_hu_r = 0.5 * (f_hu + jnp.roll(f_hu, -1)) - 0.5 * (step_dx / step_dt) * (jnp.roll(hu, -1) - hu)
            f_h_l = jnp.roll(f_h_r, 1)
            f_hu_l = jnp.roll(f_hu_r, 1)
            new_h = jnp.maximum(0.1, h - (step_dt / step_dx) * (f_h_r - f_h_l))
            new_hu = hu - (step_dt / step_dx) * (f_hu_r - f_hu_l)
            return (new_h, new_hu)

        return lax.fori_loop(0, trip, body, (h_init, hu_init), unroll=factor)

    return built(fn, init_h, init_hu, g, dx, dt)


# 6. sim_heat_crank_nicolson_step: Crank-Nicolson Implicit-Explicit Thermal Diffusion Stepping
def _crank_nicolson_factory(variant: str, trip: int, factor: int):
    n = {"n64": 64, "n128": 128, "n256": 256}[variant]
    rng = np.random.default_rng(583_856)
    init_u = jnp.asarray(np.sin(np.pi * np.linspace(0, 1, n)), dtype=jnp.float32)
    r = jnp.float32(0.25)

    def fn(u_init: jax.Array, diff_r: jax.Array):
        def body(i: int, carry: jax.Array):
            u = carry
            # Crank-Nicolson explicit RHS: (1 - r) * u[j] + 0.5 * r * (u[j+1] + u[j-1])
            rhs = (1.0 - diff_r) * u + 0.5 * diff_r * (jnp.roll(u, 1) + jnp.roll(u, -1))
            # Gauss-Seidel 4 relaxation sweeps for the implicit (1 + r) * u_next system
            def gs_sweep(sweep: int, u_gs: jax.Array):
                def row_step(j: int, u_in: jax.Array):
                    val = (rhs[j] + 0.5 * diff_r * (u_in[(j + 1) % n] + u_in[(j - 1) % n])) / (1.0 + diff_r)
                    return u_in.at[j].set(val)
                return lax.fori_loop(0, n, row_step, u_gs)
            return lax.fori_loop(0, 4, gs_sweep, u)

        return lax.fori_loop(0, trip, body, u_init, unroll=factor)

    return built(fn, init_u, r)


# 7. sim_rigid_body_quaternion: Rigid Body Rotational Kinematics via Unit Quaternion Integration
def _rigid_body_factory(variant: str, trip: int, factor: int):
    bodies = {"b4": 4, "b8": 8, "b16": 16}[variant]
    rng = np.random.default_rng(583_857)
    torques = floats(rng, trip, bodies, 3, scale=0.5)
    inertia_diag = jnp.asarray(1.0 + rng.uniform(0.5, 2.0, size=(bodies, 3)), dtype=jnp.float32)
    init_quat = jnp.zeros((bodies, 4), dtype=jnp.float32).at[:, 0].set(1.0)
    init_omega = jnp.zeros((bodies, 3), dtype=jnp.float32)
    dt = jnp.float32(0.01)

    def fn(q_init: jax.Array, w_init: jax.Array, tau_stream: jax.Array, I_diag: jax.Array, step_dt: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            q, w = carry
            tau = tau_stream[i]
            # Euler's equations: I * dw/dt + w x (I * w) = tau
            Iw = I_diag * w
            w_cross_Iw = jnp.cross(w, Iw)
            dw = (tau - w_cross_Iw) / I_diag
            w_next = w + step_dt * dw
            # Quaternion derivative: dq/dt = 0.5 * q * [0, w]
            qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            wx, wy, wz = w_next[:, 0], w_next[:, 1], w_next[:, 2]
            dq0 = -qx * wx - qy * wy - qz * wz
            dq1 = qw * wx + qy * wz - qz * wy
            dq2 = qw * wy + qz * wx - qx * wz
            dq3 = qw * wz + qx * wy - qy * wx
            dq = 0.5 * jnp.stack([dq0, dq1, dq2, dq3], axis=-1)
            q_unnorm = q + step_dt * dq
            q_next = q_unnorm / (jnp.linalg.norm(q_unnorm, axis=-1, keepdims=True) + 1e-8)
            return (q_next, w_next)

        return lax.fori_loop(0, trip, body, (q_init, w_init), unroll=factor)

    return built(fn, init_quat, init_omega, torques, inertia_diag, dt)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec("sim_verlet_symplectic", "numerical_simulation", "Velocity Verlet / Leapfrog Symplectic N-Body Integrator",
               "sim_symplectic_verlet", "Symplectic two-stage advance of nonlinear pairwise repulsive interactions",
               "fori_loop", {"n16": "16 bodies", "n32": "32 bodies", "n64": "64 bodies"}, "n32",
               _verlet_factory, 583_851, Provenance("public-inspired", "Verlet (1967) Phys. Rev.; JAX MD", "Apache-2.0", "Symplectic velocity Verlet integrator"),
               tier="full", related_group="sim_verlet_eval"),
    CorpusSpec("sim_reaction_diffusion_1d", "numerical_simulation", "Gray-Scott Coupled Reaction-Diffusion PDE Ring",
               "sim_reaction_diffusion", "Couples two nonlinear PDE fields with cubic kinetics u*v^2",
               "fori_loop", {"g64": "grid 64", "g128": "grid 128", "g256": "grid 256"}, "g128",
               _reaction_diff_factory, 583_852, Provenance("public-inspired", "Pearson (1993) Science; JAX-CFD", "Apache-2.0", "Gray-Scott reaction diffusion system"),
               tier="full", related_group="sim_gray_scott"),
    CorpusSpec("sim_finite_volume_burgers", "numerical_simulation", "Godunov Upwind Non-Linear Burgers' Shock Capturing Solver",
               "sim_burgers_godunov", "Solves non-linear conservation law with Godunov numerical shock flux",
               "fori_loop", {"n64": "grid 64", "n128": "grid 128", "n256": "grid 256"}, "n128",
               _burgers_factory, 583_853, Provenance("public-inspired", "Godunov (1959); LeVeque Finite Volume", "Public Domain", "Godunov finite volume shock solver"),
               tier="smoke", related_group="sim_burgers"),
    CorpusSpec("sim_pbd_cloth_spring", "numerical_simulation", "Position-Based Dynamics Distance Constraint Projection",
               "sim_pbd_constraints", "Geometric constraint projection along position gradients",
               "fori_loop", {"n32": "32 vertices", "n64": "64 vertices", "n128": "128 vertices"}, "n64",
               _pbd_factory, 583_854, Provenance("public-inspired", "Muller et al. (2007) J. Vis. Commun. Image R.", "Public Domain", "Position based dynamics distance projection"),
               tier="full", related_group="sim_pbd"),
    CorpusSpec("sim_shallow_water_1d", "numerical_simulation", "1D Saint-Venant Shallow Water Equations with Lax-Friedrichs Flux",
               "sim_shallow_water", "Hyperbolic conservation law system with Lax-Friedrichs interface flux",
               "fori_loop", {"g64": "grid 64", "g128": "grid 128", "g256": "grid 256"}, "g128",
               _shallow_water_factory, 583_855, Provenance("public-inspired", "Toro (2001) Shock-Capturing Methods", "Public Domain", "1D Saint-Venant shallow water equations"),
               tier="full", related_group="sim_shallow_eval"),
    CorpusSpec("sim_heat_crank_nicolson_step", "numerical_simulation", "Crank-Nicolson Implicit-Explicit Thermal Diffusion Stepping",
               "sim_crank_nicolson", "Implicit-explicit thermal diffusion with iterative relaxation solver",
               "fori_loop", {"n64": "grid 64", "n128": "grid 128", "n256": "grid 256"}, "n128",
               _crank_nicolson_factory, 583_856, Provenance("public-inspired", "Crank & Nicolson (1947) Proc. Camb. Phil. Soc.", "Public Domain", "Crank-Nicolson parabolic PDE solver"),
               tier="full", related_group="sim_thermal"),
    CorpusSpec("sim_rigid_body_quaternion", "numerical_simulation", "Rigid Body Rotational Kinematics via Unit Quaternion Integration",
               "sim_rigid_quaternion", "Euler rotational equations coupled with quaternion derivative and normalization",
               "fori_loop", {"b4": "4 bodies", "b8": "8 bodies", "b16": "16 bodies"}, "b8",
               _rigid_body_factory, 583_857, Provenance("public-inspired", "Baraff (1997) SIGGRAPH Course Notes", "Public Domain", "Rigid body quaternion dynamics"),
               tier="full", related_group="sim_rigid_body"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
