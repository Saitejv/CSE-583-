"""Domain 6: Application kernels workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats
from benchmarks.corpus.spec import CORPUS_TRIP_COUNTS, CorpusSpec, Provenance
from jax_unroll.auto_loop import precompute_fori_loop


# 1. app_rigid_quaternion (train): Rigid body quaternion rotation integration
def _rigid_quaternion_factory(variant: str, trip: int, factor: int):
    bodies = 1 if variant == "small" else 4
    rng = np.random.default_rng(583_601)
    # State: quaternion (bodies, 4), angular_velocity (bodies, 3)
    q_init_np = np.zeros((bodies, 4), dtype=np.float32)
    q_init_np[:, 0] = 1.0  # identity quaternion
    init_q = jnp.asarray(q_init_np)
    init_omega = floats(rng, bodies, 3, scale=0.1)
    
    base_torques = floats(rng, 1024, bodies, 3, scale=0.05)
    torques = base_torques[:trip]
    dt = jnp.float32(0.01)

    def fn(q: jax.Array, omega: jax.Array, tau_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            quat, om = carry
            tau = tau_stream[i]
            # Angular velocity update: omega += dt * torque (simplified, unit inertia)
            new_om = om + dt * tau
            # Quaternion derivative: dq/dt = 0.5 * q * omega_quat
            # omega_quat = (0, omega_x, omega_y, omega_z)
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            ox, oy, oz = new_om[:, 0], new_om[:, 1], new_om[:, 2]
            dw = 0.5 * (-x * ox - y * oy - z * oz)
            dx = 0.5 * (w * ox + y * oz - z * oy)
            dy = 0.5 * (w * oy + z * ox - x * oz)
            dz = 0.5 * (w * oz + x * oy - y * ox)
            new_q = jnp.stack([w + dt * dw, x + dt * dx, y + dt * dy, z + dt * dz], axis=1)
            # Re-normalize quaternion
            norm = jnp.linalg.norm(new_q, axis=1, keepdims=True) + 1e-8
            new_q = new_q / norm
            return (new_q, new_om)

        return lax.fori_loop(0, trip, body, (q, omega), unroll=factor)

    return built(fn, init_q, init_omega, torques)


# 2. app_coordinate_descent (train): Cyclic coordinate descent with soft-thresholding
def _coordinate_descent_factory(variant: str, trip: int, factor: int):
    features = 16 if variant == "small" else 64
    rng = np.random.default_rng(583_602)
    init_beta = jnp.zeros((features,), dtype=jnp.float32)
    
    # Pre-computed correlation vector and diagonal Hessian
    base_corr = floats(rng, 1024, features, scale=1.0)
    corr_stream = base_corr[:trip]
    diag_h = jnp.abs(floats(rng, features, scale=1.0)) + 0.1
    lam = jnp.float32(0.1)

    def fn(beta: jax.Array, corr_in: jax.Array, hess_diag: jax.Array):
        def body(i: int, b: jax.Array):
            j = i % features
            corr = corr_in[i]
            r_j = corr[j] - hess_diag[j] * b[j]
            # Soft-thresholding
            soft = jnp.sign(r_j) * jnp.maximum(0.0, jnp.abs(r_j) - lam)
            new_bj = soft / hess_diag[j]
            return b.at[j].set(new_bj)

        return lax.fori_loop(0, trip, body, beta, unroll=factor)

    return built(fn, init_beta, corr_stream, diag_h)


# 3. app_molecular_verlet (train): Velocity-Verlet for Lennard-Jones particle chain
def _molecular_verlet_factory(variant: str, trip: int, factor: int):
    particles = 8 if variant == "small" else 32
    rng = np.random.default_rng(583_603)
    init_pos = jnp.linspace(1.0, float(particles), particles, dtype=jnp.float32)
    init_vel = floats(rng, particles, scale=0.05)
    
    base_perturbations = floats(rng, 1024, particles, scale=0.01)
    perturbations = base_perturbations[:trip]
    dt = jnp.float32(0.005)
    sigma = jnp.float32(1.0)
    epsilon = jnp.float32(0.1)

    def fn(pos: jax.Array, vel: jax.Array, pert_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            x, v = carry
            # Nearest-neighbor LJ force
            dx_right = jnp.roll(x, -1) - x
            dx_left = x - jnp.roll(x, 1)
            
            def lj_force(dr):
                r = jnp.abs(dr) + 0.3
                r6 = (sigma / r) ** 6
                return 24.0 * epsilon * (2.0 * r6 * r6 - r6) / r * jnp.sign(dr)
            
            f = lj_force(dx_right) - lj_force(dx_left) + pert_stream[i]
            # Verlet: half-step velocity, full-step position, half-step velocity
            v_half = v + 0.5 * dt * f
            new_x = x + dt * v_half
            # Recompute force at new position
            dx_right2 = jnp.roll(new_x, -1) - new_x
            dx_left2 = new_x - jnp.roll(new_x, 1)
            f2 = lj_force(dx_right2) - lj_force(dx_left2) + pert_stream[i]
            new_v = v_half + 0.5 * dt * f2
            return (new_x, new_v * 0.999)  # slight damping

        return lax.fori_loop(0, trip, body, (pos, vel), unroll=factor)

    return built(fn, init_pos, init_vel, perturbations)


# 4. app_robot_kinematic_chain (train): Forward kinematics SO(3) rotation chain
def _robot_kinematic_factory(variant: str, trip: int, factor: int):
    joints = 6 if variant == "small" else 18
    rng = np.random.default_rng(583_604)
    init_rot = jnp.eye(3, dtype=jnp.float32)
    init_pos = jnp.zeros((3,), dtype=jnp.float32)
    
    # Joint angles stream
    base_angles = floats(rng, 1024, joints, scale=0.1)
    angles = base_angles[:trip]
    # Link lengths
    link_len = jnp.ones((joints,), dtype=jnp.float32) * 0.5

    def fn(rot: jax.Array, pos: jax.Array, ang_stream: jax.Array, links: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            r, p = carry
            ang = ang_stream[i]
            # For each step, apply first joint's rotation (cycle through joints)
            j = i % joints
            theta = ang[j]
            c, s = jnp.cos(theta), jnp.sin(theta)
            # Rotation about z-axis (simplification for chain)
            rz = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            new_r = jnp.dot(r, rz)
            # Translate along local x-axis by link length
            new_p = p + new_r[:, 0] * links[j]
            return (new_r, new_p)

        return lax.fori_loop(0, trip, body, (rot, pos), unroll=factor)

    return built(fn, init_rot, init_pos, angles, link_len)


# 5. eval_app_rk4_lorenz_step (eval): 4th-order Runge-Kutta Lorenz system integration
def _rk4_lorenz_factory(variant: str, trip: int, factor: int):
    systems = 2 if variant == "small" else 8
    rng = np.random.default_rng(583_605)
    # State: (systems, 3) for x, y, z
    init_state = floats(rng, systems, 3, scale=1.0)
    dt = jnp.float32(0.005)
    sigma_l = jnp.float32(10.0)
    rho = jnp.float32(28.0)
    beta = jnp.float32(8.0 / 3.0)
    
    base_perturbations = floats(rng, 1024, systems, 3, scale=0.001)
    perts = base_perturbations[:trip]

    def fn(state: jax.Array, pert_stream: jax.Array):
        def lorenz(s):
            x, y, z = s[:, 0], s[:, 1], s[:, 2]
            dx = sigma_l * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return jnp.stack([dx, dy, dz], axis=1)

        def body(i: int, s: jax.Array):
            k1 = lorenz(s)
            k2 = lorenz(s + 0.5 * dt * k1)
            k3 = lorenz(s + 0.5 * dt * k2)
            k4 = lorenz(s + dt * k3)
            new_s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return new_s + pert_stream[i]

        return lax.fori_loop(0, trip, body, state, unroll=factor)

    return built(fn, init_state, perts)


# 6. eval_app_trajectory_shooting (eval): Spacecraft trajectory shooting under gravity
def _trajectory_shooting_factory(variant: str, trip: int, factor: int):
    vehicles = 1 if variant == "small" else 4
    rng = np.random.default_rng(583_606)
    # State: position (vehicles, 3), velocity (vehicles, 3)
    init_pos = floats(rng, vehicles, 3, scale=5.0)
    init_vel = floats(rng, vehicles, 3, scale=0.5)
    
    base_thrust = floats(rng, 1024, vehicles, 3, scale=0.02)
    thrusts = base_thrust[:trip]
    dt = jnp.float32(0.01)
    mu = jnp.float32(1.0)  # gravitational parameter

    def fn(pos: jax.Array, vel: jax.Array, thrust_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            p, v = carry
            r = jnp.linalg.norm(p, axis=1, keepdims=True) + 0.5
            # Central body gravitational acceleration
            a_grav = -mu * p / (r ** 3)
            a_total = a_grav + thrust_stream[i]
            new_v = v + dt * a_total
            new_p = p + dt * new_v
            return (new_p, new_v)

        return lax.fori_loop(0, trip, body, (pos, vel), unroll=factor)

    return built(fn, init_pos, init_vel, thrusts)


# 7. eval_app_ct_backprojection_step (eval): CT filtered backprojection ray accumulation
def _ct_backprojection_factory(variant: str, trip: int, factor: int, *, precompute=False):
    dim = 16 if variant == "small" else 32
    rng = np.random.default_rng(583_607)
    init_image = jnp.zeros((dim, dim), dtype=jnp.float32)
    
    # Sinogram: projection data at different angles
    base_sino = floats(rng, 1024, dim, scale=1.0)
    sino = base_sino[:trip]
    
    # Pre-computed cos/sin angle sequence
    base_angles = np.linspace(0, np.pi, 1024, endpoint=False, dtype=np.float32)
    cos_a = jnp.asarray(np.cos(base_angles[:trip]))
    sin_a = jnp.asarray(np.sin(base_angles[:trip]))
    
    # Grid coordinates centered at origin
    coords = jnp.linspace(-1.0, 1.0, dim, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(coords, coords)

    def fn(image: jax.Array, sinogram: jax.Array, ca: jax.Array, sa: jax.Array, gx: jax.Array, gy: jax.Array):
        def project(i: int):
            proj = sinogram[i]
            c, s = ca[i], sa[i]
            # Compute projection coordinate t for each pixel
            t = c * gx + s * gy
            # Map t to detector index (linear interpolation)
            t_idx = (t + 1.0) * 0.5 * (dim - 1)
            t_idx = jnp.clip(t_idx, 0.0, dim - 1.0)
            idx_lo = jnp.floor(t_idx).astype(jnp.int32)
            idx_hi = jnp.minimum(idx_lo + 1, dim - 1)
            frac = t_idx - jnp.floor(t_idx)
            # Linear interpolation of projection data
            return (1.0 - frac) * proj[idx_lo] + frac * proj[idx_hi]

        if precompute:
            # Keep the batched projection and the scan over full blocks free
            # from the tail. The accumulator still sees angles in order.
            full = trip - trip % factor
            if full:
                image = precompute_fori_loop(0, full, project, jnp.add, image, unroll=factor)
            if full < trip:
                image = precompute_fori_loop(full, trip, project, jnp.add, image, unroll=True)
            return image
        return lax.fori_loop(0, trip, lambda i, img: img + project(i), image, unroll=factor)

    return built(fn, init_image, sino, cos_a, sin_a, xx, yy)


# 8. eval_app_epidemic_seir_rk4 (eval): SEIR compartmental ODE solved via RK4
def _epidemic_seir_rk4_factory(variant: str, trip: int, factor: int):
    regions = 2 if variant == "small" else 8
    rng = np.random.default_rng(583_608)
    # State: (regions, 4) for S, E, I, R fractions
    init_state_np = np.zeros((regions, 4), dtype=np.float32)
    init_state_np[:, 0] = 0.99  # S
    init_state_np[:, 1] = 0.005  # E
    init_state_np[:, 2] = 0.005  # I
    init_state = jnp.asarray(init_state_np)
    
    dt = jnp.float32(0.1)
    base_beta_var = rng.uniform(0.95, 1.05, size=(1024, regions)).astype(np.float32)
    beta_vars = jnp.asarray(base_beta_var[:trip])
    
    beta0 = jnp.float32(0.3)
    sigma_e = jnp.float32(0.2)  # 1/incubation
    gamma = jnp.float32(0.1)    # 1/infectious

    def fn(state: jax.Array, beta_v: jax.Array):
        def seir_deriv(s, beta_mult):
            ss, ee, ii, rr = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
            beta = beta0 * beta_mult
            ds = -beta * ss * ii
            de = beta * ss * ii - sigma_e * ee
            di = sigma_e * ee - gamma * ii
            dr = gamma * ii
            return jnp.stack([ds, de, di, dr], axis=1)

        def body(i: int, s: jax.Array):
            bv = beta_v[i]
            k1 = seir_deriv(s, bv)
            k2 = seir_deriv(s + 0.5 * dt * k1, bv)
            k3 = seir_deriv(s + 0.5 * dt * k2, bv)
            k4 = seir_deriv(s + dt * k3, bv)
            new_s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # Clamp to valid fractions
            return jnp.clip(new_s, 0.0, 1.0)

        return lax.fori_loop(0, trip, body, state, unroll=factor)

    return built(fn, init_state, beta_vars)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        name="app_rigid_quaternion",
        family="application_kernels",
        description="Rigid body rotational mechanics with quaternion angular velocity integration",
        mechanism="quaternion_angular_velocity_integration",
        structural_difference="Quaternion multiplication derivative, angular velocity carry, and unit normalization",
        loop_kind="fori_loop",
        shape_variants={
            "small": "bodies=1 (q 4 floats + omega 3 floats = 28 B)",
            "large": "bodies=4 (q 16 floats + omega 12 floats = 112 B)",
        },
        primary_variant="small",
        factory=_rigid_quaternion_factory,
        seed=583_601,
        provenance=Provenance(
            kind="original",
            source="Adapted from sim_rigid_body_quaternion historical lineage",
            license="MIT",
            adaptation="Expanded to multi-body variants (1, 4) with torque stream input",
        ),
        related_group="rigid_quaternion",
    ),
    CorpusSpec(
        name="app_coordinate_descent",
        family="application_kernels",
        description="Cyclic coordinate descent with soft-thresholding for L1 regularization",
        mechanism="cyclic_coordinate_descent_soft_threshold",
        structural_difference="Single-coordinate update with soft-thresholding operator and cyclic index",
        loop_kind="fori_loop",
        shape_variants={
            "small": "features=16 (64 B)",
            "large": "features=64 (256 B)",
        },
        primary_variant="small",
        factory=_coordinate_descent_factory,
        seed=583_602,
        provenance=Provenance(
            kind="original",
            source="Adapted from opt_coordinate_descent historical lineage",
            license="MIT",
            adaptation="Standardized to dual feature dimensions with deterministic correlation stream",
        ),
        related_group="coordinate_descent",
    ),
    CorpusSpec(
        name="app_molecular_verlet",
        family="application_kernels",
        description="Symplectic velocity-Verlet integrator for Lennard-Jones particle chain",
        mechanism="velocity_verlet_lennard_jones_chain",
        structural_difference="Symplectic half-step velocity-position-velocity with LJ nearest-neighbor force evaluation",
        loop_kind="fori_loop",
        shape_variants={
            "small": "particles=8 (pos 32 B + vel 32 B = 64 B)",
            "large": "particles=32 (pos 128 B + vel 128 B = 256 B)",
        },
        primary_variant="small",
        factory=_molecular_verlet_factory,
        seed=583_603,
        provenance=Provenance(
            kind="public-inspired",
            source="Verlet, L. (1967). Computer experiments on classical fluids. Phys. Rev. 159(1)",
            license="Academic reference / public domain",
            adaptation="1D chain Lennard-Jones molecular dynamics with symplectic integration",
        ),
        related_group="molecular_verlet",
    ),
    CorpusSpec(
        name="app_robot_kinematic_chain",
        family="application_kernels",
        description="Forward kinematics iterating through joint angles accumulating SO(3) rotation frames",
        mechanism="forward_kinematics_rotation_chain",
        structural_difference="Sequential 3x3 rotation matrix multiplication and link translation accumulation",
        loop_kind="fori_loop",
        shape_variants={
            "small": "joints=6 (rot 36 B + pos 12 B = 48 B)",
            "large": "joints=18 (rot 36 B + pos 12 B = 48 B, link_len 72 B)",
        },
        primary_variant="small",
        factory=_robot_kinematic_factory,
        seed=583_604,
        provenance=Provenance(
            kind="public-inspired",
            source="Denavit, J., & Hartenberg, R. S. (1955). A kinematic notation for lower-pair mechanisms. J. Appl. Mech.",
            license="Academic / public domain",
            adaptation="DH-convention simplified z-axis rotation chain for serial manipulator",
        ),
        related_group="robot_kinematic_chain",
    ),
    CorpusSpec(
        name="eval_app_rk4_lorenz_step",
        family="eval_application_kernels",
        description="4th-order Runge-Kutta integration of 3D chaotic Lorenz system",
        mechanism="rk4_lorenz_attractor_integration",
        structural_difference="Four coupled derivative evaluations (k1-k4) with 3D nonlinear mixing",
        loop_kind="fori_loop",
        shape_variants={
            "small": "systems=2 (state 2x3 = 24 B)",
            "large": "systems=8 (state 8x3 = 96 B)",
        },
        primary_variant="small",
        factory=_rk4_lorenz_factory,
        seed=583_605,
        provenance=Provenance(
            kind="public-inspired",
            source="Lorenz, E. N. (1963). Deterministic nonperiodic flow. J. Atmos. Sci. 20(2)",
            license="Public domain / academic",
            adaptation="Multi-system parallel Lorenz attractor integration with perturbation stream",
        ),
        related_group="eval_rk4_lorenz",
    ),
    CorpusSpec(
        name="eval_app_trajectory_shooting",
        family="eval_application_kernels",
        description="Direct shooting spacecraft trajectory under gravitational central body acceleration",
        mechanism="trajectory_shooting_central_gravity",
        structural_difference="Inverse-square gravitational force evaluation and thrust vector superposition",
        loop_kind="fori_loop",
        shape_variants={
            "small": "vehicles=1 (pos 12 B + vel 12 B = 24 B)",
            "large": "vehicles=4 (pos 48 B + vel 48 B = 96 B)",
        },
        primary_variant="small",
        factory=_trajectory_shooting_factory,
        seed=583_606,
        provenance=Provenance(
            kind="public-inspired",
            source="Betts, J. T. (1998). Survey of numerical methods for trajectory optimization. JGCD 21(2)",
            license="Academic / AIAA reference",
            adaptation="Single-segment forward shooting integrator with central body gravity",
        ),
        related_group="eval_trajectory_shooting",
    ),
    CorpusSpec(
        name="eval_app_ct_backprojection_step",
        family="eval_application_kernels",
        description="Filtered backprojection ray accumulation for CT reconstruction",
        mechanism="ct_filtered_backprojection_accumulation",
        structural_difference="Trigonometric ray coordinate mapping, linear interpolation, and 2D grid accumulation",
        loop_kind="fori_loop",
        shape_variants={
            "small": "image=16x16 (1 KB)",
            "large": "image=32x32 (4 KB)",
        },
        primary_variant="small",
        factory=_ct_backprojection_factory,
        seed=583_607,
        provenance=Provenance(
            kind="public-inspired",
            source="Kak, A. C., & Slaney, M. (2001). Principles of computerized tomographic imaging. SIAM",
            license="Academic standard",
            adaptation="Radon transform backprojection accumulation over streaming projection angles",
        ),
        related_group="eval_ct_backprojection",
    ),
    CorpusSpec(
        name="eval_app_epidemic_seir_rk4",
        family="eval_application_kernels",
        description="SEIR compartmental disease transmission ODE solved via RK4",
        mechanism="seir_compartmental_rk4_integration",
        structural_difference="Four SEIR derivative evaluations (k1-k4) with bilinear transmission coupling",
        loop_kind="fori_loop",
        shape_variants={
            "small": "regions=2 (state 2x4 = 32 B)",
            "large": "regions=8 (state 8x4 = 128 B)",
        },
        primary_variant="small",
        factory=_epidemic_seir_rk4_factory,
        seed=583_608,
        provenance=Provenance(
            kind="public-inspired",
            source="Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics. Proc. R. Soc. A",
            license="Public domain / academic",
            adaptation="Multi-region SEIR model with time-varying transmission rate",
        ),
        related_group="eval_epidemic_seir",
    ),
)
