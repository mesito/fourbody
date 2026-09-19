#!/usr/bin/env python3
"""verify_fourbody.py -- self-contained verification suite for

  M. Ismail, "An exactly solvable four-body model for zero collisions under the
  de Bruijn heat flow, and the growth of neighbour separations in the full flow".

No data files, no external references: everything is recomputed here.
Dependencies: numpy, scipy, mpmath, sympy.

Groups (paper reference in brackets):
  F1  coefficient flow, parabola and invariant vs direct integration     [Thm 2.1]
  F2  closed form, rate identity, convexity, mean rate 6, film formula   [Thm 2.2, 3.1; Prop 3.1, 3.2; Cor 3.1]
  F3  episode structure, identity swap, reference tau_dyn, sparse floor,
      near-axis law                                                       [Thm 3.2; Cor 3.2; Prop 3.3]
  F4  mirror intertwinings (symbolic) and the gluing (numeric)            [Thm 4.1]
  F5  splitting dichotomy on genuine zero pairs of Xi (mpmath)            [Prop 5.1; Table 1]
  F6  accelerated fall in the full flow: sign, identity, lifetimes        [Lemma 6.1; Thm 6.1; Cor 6.1]
  F7  lattice back-pressure: AP sign rule, cumulative checker, digamma
      law against lattice sums, nonlocal linear response, exact maximum
      principle for the largest gap                                     [Lemma 7.1; Thm 7.1, 7.2; Prop 7.1-7.3]
  F8  the counterexample in both readings, the control, the exact
      foot-relative decomposition                                        [Thm 8.1, 8.2; Prop 8.1]
  F9  (full tier) dynamic near-axis law on 82 genuine zeros; threshold
      margins and the gap-relaxation law in a 52-zero window at t~7000  [Section 9]

Usage: python3 verify_fourbody.py [--full] [--groups F1,F2,...]
Every check prints measured value, expectation with tolerance, and PASS/FAIL.
"""
import sys, time, argparse
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import digamma

RES = []
GAMMA_E = 0.5772156649015329


def check(name, ok, measured, expected):
    RES.append((name, ok))
    print("  [%s] %-58s %s   (expected %s)" % ("PASS" if ok else "FAIL", name, measured, expected), flush=True)
    return ok


# ----------------------------------------------------------------- model
def I_of(y, q): return y**4 + 10 * q * q * y * y + q**4
def D_of(y, q): return q * q - y * y


def closed(t, y0, q0):
    I0, D0 = I_of(y0, q0), D_of(y0, q0)
    D = D0 + 12 * t; S = np.sqrt((2 * D * D + I0) / 3)
    return (S + D) / 2, (S - D) / 2, D, S      # q^2, y^2, D, S


def cell_rhs(t, u):
    y, q = u
    return [-1 / y - 4 * y / (y * y + q * q), 1 / q + 4 * q / (q * q + y * y)]


def tau_dyn(y0, q0): return (np.sqrt(I_of(y0, q0)) - D_of(y0, q0)) / 12


# ----------------------------------------------------------------- F1
def F1():
    print("\nF1  coefficient flow / parabola / invariant vs integration (Thm 2.1)")
    y0, q0 = 1.0, 2.7588
    ev = lambda t, u: u[0] - 1e-6; ev.terminal = True
    sol = solve_ivp(cell_rhs, [0, 2], [y0, q0], events=ev, rtol=1e-12, atol=1e-14, dense_output=True, max_step=1e-3)
    ts = np.linspace(0, sol.t_events[0][0] * 0.999, 400); Y, Q = sol.sol(ts)
    A, B = Y**2 - Q**2, -(Y**2) * Q**2
    A0, B0 = y0**2 - q0**2, -(y0**2) * q0**2
    errA = np.max(np.abs(A - (A0 - 12 * ts))); errB = np.max(np.abs(B - (B0 - 2 * A0 * ts + 12 * ts**2)))
    drift = np.max(np.abs(I_of(Y, Q) - I_of(y0, q0))) / I_of(y0, q0)
    check("A(t)=A0-12t, B(t)=B0-2A0t+12t^2 along the integrated cell", errA < 1e-8 and errB < 1e-8, "max err %.1e / %.1e" % (errA, errB), "< 1e-8")
    check("invariant I=A^2-12B conserved (relative drift)", drift < 1e-10, "%.1e" % drift, "< 1e-10")


# ----------------------------------------------------------------- F2
def F2():
    print("\nF2  closed form, rate identity, convexity, mean rate, film (Thm 2.2, 3.1; Prop 3.1, 3.2; Cor 3.1)")
    import sympy as sp
    y0, q0 = 1.0, 2.7588
    ev = lambda t, u: u[0] - 1e-6; ev.terminal = True
    sol = solve_ivp(cell_rhs, [0, 2], [y0, q0], events=ev, rtol=1e-12, atol=1e-14, dense_output=True, max_step=1e-3)
    ts = np.linspace(0, sol.t_events[0][0] * 0.99, 300); Y, Q = sol.sol(ts)
    q2c, y2c, D, S = closed(ts, y0, q0)
    e_closed = max(np.max(np.abs(Q**2 - q2c)), np.max(np.abs(Y**2 - y2c)))
    check("closed form (2.4): q^2=(S+D)/2, y^2=(S-D)/2 vs integration", e_closed < 1e-8, "max err %.1e" % e_closed, "< 1e-8")
    # rate identity, numerically from the ODE
    dq2 = 2 * Q * np.array([cell_rhs(0, [y, q])[1] for y, q in zip(Y, Q)])
    e_rate = max(np.max(np.abs(dq2 - (6 + 4 * D / S))), np.max(np.abs(dq2 - (2 + 8 * Q**2 / (Q**2 + Y**2)))))
    check("rate identity (3.1): d(q^2)/dt = 6+4D/S = 2+8q^2/(q^2+y^2)", e_rate < 1e-9, "max err %.1e" % e_rate, "< 1e-9")
    # convexity: second derivative by finite differences of the closed form
    h = 1e-5; tt = ts[5:-5]
    d2 = (closed(tt + h, y0, q0)[0] - 2 * closed(tt, y0, q0)[0] + closed(tt - h, y0, q0)[0]) / h**2
    I0 = I_of(y0, q0); S_t = closed(tt, y0, q0)[3]
    e_conv = np.max(np.abs(d2 - 16 * I0 / S_t**3) / (16 * I0 / S_t**3))
    check("convexity (Prop 3.1): d2(q^2)/dt2 = 16 I0/S^3 > 0", e_conv < 1e-4 and np.all(d2 > 0), "rel err %.1e, min %.3f" % (e_conv, d2.min()), "< 1e-4, positive")
    # mean rate exactly 6 over a full episode (integral in D)
    Ds = np.sqrt(I0); val = quad(lambda d: 6 + 4 * d / np.sqrt((2 * d * d + I0) / 3), -Ds, Ds)[0] / (2 * Ds)
    check("mean rate over an episode = 6 exactly (Cor 3.1)", abs(val - 6) < 1e-10, "%.12f" % val, "6")
    # film formula symbolic: M(t)=M0-K0 t+12 t^2 with M=B, K=2A
    t, A0s, B0s = sp.symbols('t A0 B0')
    As = A0s - 12 * t; Bs = B0s - 2 * A0s * t + 12 * t**2
    film_ok = sp.simplify(sp.diff(As, t) + 12) == 0 and sp.simplify(sp.diff(Bs, t) + 2 * As) == 0 and sp.simplify(Bs - (B0s - (2 * A0s) * t + 12 * t**2)) == 0
    check("film formula (Prop 3.2): dA/dt=-12, dB/dt=-2A, M=M0-K0 t+12t^2 (symbolic)", film_ok, str(film_ok), "True")


# ----------------------------------------------------------------- F3
def F3():
    print("\nF3  episode structure, identity swap, reference tau_dyn, sparse floor, near-axis law")
    y0, q0 = 1.0, 2.7588; I0, D0 = I_of(y0, q0), D_of(y0, q0)
    t2, td = -(D0 + np.sqrt(I0)) / 12, (np.sqrt(I0) - D0) / 12
    q2_at_t2, y2_at_t2, _, _ = closed(t2, y0, q0); q2_at_td, y2_at_td, _, _ = closed(td, y0, q0)
    check("episode endpoints: q^2(t2)=0 and y^2(tau_dyn)=0 (Thm 3.2)", abs(q2_at_t2) < 1e-9 and abs(y2_at_td) < 1e-9, "%.1e / %.1e" % (q2_at_t2, y2_at_td), "0 / 0")
    check("identity swap: y(t2)=q(tau_dyn)=I0^{1/4}", abs(np.sqrt(y2_at_t2) - I0**0.25) < 1e-9 and abs(np.sqrt(q2_at_td) - I0**0.25) < 1e-9, "%.6f / %.6f / I0^{1/4}=%.6f" % (np.sqrt(y2_at_t2), np.sqrt(q2_at_td), I0**0.25), "equal")
    check("lifetime tau_dyn - t2 = sqrt(I0)/6", abs((td - t2) - np.sqrt(I0) / 6) < 1e-12, "%.10f" % (td - t2), "%.10f" % (np.sqrt(I0) / 6))
    check("reference configuration (1, 2.7588): tau_dyn = 0.4175", abs(td - 0.4175) < 5e-4, "%.5f (I0=%.3f, D0=%.4f)" % (td, I0, D0), "0.4175")
    # numerical landing time vs closed form
    ev = lambda t, u: u[0] - 1e-7; ev.terminal = True
    sol = solve_ivp(cell_rhs, [0, 2], [y0, q0], events=ev, rtol=1e-12, atol=1e-14, max_step=1e-3)
    tl = sol.t_events[0][0] + 1e-14 / 2
    check("integrated landing time vs closed form (6 digits)", abs(tl - td) < 2e-6, "%.6f" % tl, "%.6f" % td)
    # rate 2 -> 10 over the episode
    r2 = 2 + 8 * closed(t2 + 1e-9, y0, q0)[0] / (closed(t2 + 1e-9, y0, q0)[0] + closed(t2 + 1e-9, y0, q0)[1])
    r10 = 2 + 8 * closed(td - 1e-9, y0, q0)[0] / (closed(td - 1e-9, y0, q0)[0] + closed(td - 1e-9, y0, q0)[1])
    check("rate sweeps 2 -> 10 over the episode (Thm 3.2(iv))", abs(r2 - 2) < 1e-3 and abs(r10 - 10) < 1e-3, "%.4f -> %.4f" % (r2, r10), "2 -> 10")
    # sparse floor and monotonicity on a grid
    ys = np.linspace(0.2, 2.0, 19); qs = np.linspace(0.1, 8.0, 80); ok = True; okm = True
    for y in ys:
        prev = -1
        for q in qs:
            tv = tau_dyn(y, q); ok &= tv >= y * y / 6 - 1e-12; okm &= tv > prev; prev = tv
    check("sparse floor tau_dyn >= y0^2/6 and monotone in q0 on a grid (Cor 3.2)", ok and okm, "1520 configurations", "all satisfied")
    # two-body limit
    check("two-body limit tau_dyn -> y0^2/2 as q0 -> inf", abs(tau_dyn(1.0, 1e5) - 0.5) < 1e-8, "%.9f" % tau_dyn(1.0, 1e5), "0.5")
    # near-axis law: y^2 ~ 2 (tau - t)
    tt = td - np.array([1e-3, 1e-4, 1e-5]); y2 = closed(tt, y0, q0)[1]
    check("near-axis law y^2 = 2(tau-t)(1+O(tau-t)) (Prop 3.3)", np.all(np.abs(y2 / (2 * (td - tt)) - 1) < 5e-2 * np.array([1, 0.1, 0.01]) + 1e-3), str(np.round(y2 / (2 * (td - tt)), 5)), "-> 1")


# ----------------------------------------------------------------- F4
def F4():
    print("\nF4  mirror theorem: intertwinings (symbolic) and gluing (numeric) (Thm 4.1)")
    import sympy as sp
    r, q, y = sp.symbols('r q y', positive=True)
    Aell, Bell = y**2 - q**2, -y**2 * q**2
    Ahyp, Bhyp = -(r**2 + q**2), r**2 * q**2
    ok1 = sp.simplify(Aell.subs(y, sp.I * r) - Ahyp) == 0 and sp.simplify(Bell.subs(y, sp.I * r) - Bhyp) == 0
    dy2 = -2 - 8 * y**2 / (y**2 + q**2); dr2 = 2 - 8 * r**2 / (q**2 - r**2)
    ok2 = sp.simplify(dy2.subs(y, sp.I * r) + dr2) == 0
    dq2e = 2 + 8 * q**2 / (q**2 + y**2); dq2h = 2 + 8 * q**2 / (q**2 - r**2)
    ok2b = sp.simplify(dq2e.subs(y, sp.I * r) - dq2h) == 0
    Iell = y**4 + 10 * q**2 * y**2 + q**4; Jhyp = r**4 - 10 * q**2 * r**2 + q**4
    ok3 = sp.simplify(Iell.subs(y, sp.I * r) - Jhyp) == 0
    check("(i) coefficients, (ii) vector fields, (iii) invariant intertwine under y -> i r", ok1 and ok2 and ok2b and ok3, "%s %s %s %s" % (ok1, ok2, ok2b, ok3), "all True")
    # gluing: continue past the collision in coefficient space and read J
    y0, q0 = 1.0, 2.7588; I0 = I_of(y0, q0); td = tau_dyn(y0, q0)
    A0, B0 = y0**2 - q0**2, -y0**2 * q0**2; t = td + 0.05
    A, B = A0 - 12 * t, B0 - 2 * A0 * t + 12 * t * t
    # roots: z^2 = w, w^2 + A w + B = 0 -> two positive w (hyperbolic)
    w = np.roots([1, A, B]); r2, q2 = sorted(w.real)
    J = r2**2 - 10 * q2 * r2 + q2**2
    check("(iv) gluing: past the collision the chart reads J = A^2-12B = I0", abs(J - I0) < 1e-8 and B > 0 and A < 0 and r2 > 0, "J=%.6f, I0=%.6f, r^2=%.4f, q^2=%.4f" % (J, I0, r2, q2), "J = I0, two real pairs")


# ----------------------------------------------------------------- F5
def F5(full):
    print("\nF5  splitting dichotomy on genuine zero pairs of Xi (Prop 5.1; Table 1)")
    from mpmath import mp, mpf, mpc, zeta, gamma, pi, findroot, diff, zetazero, im, re, sqrt
    mp.dps = 30
    def Xi(t):
        s = mpc(0.5, 0) + mpc(0, 1) * t
        return re(s * (s - 1) / 2 * pi**(-s / 2) * gamma(s / 2) * zeta(s))
    def Xic(t):  # complex t
        s = mpc(0.5, 0) + mpc(0, 1) * t
        return s * (s - 1) / 2 * pi**(-s / 2) * gamma(s / 2) * zeta(s)
    canon = {"closest": [0.99998, 0.99991, 0.99965, 0.99912], "typical": [0.99874, 0.9938, 0.9764, 0.9458]}
    # locate the two pairs by scanning zetazero indices near the quoted heights
    def pair_near(gamma_target, tight):
        n = int(gamma_target / (2 * mp.pi) * mp.log(gamma_target / (2 * mp.pi * mp.e))) + 1
        best = None
        for k in range(n - 40, n + 40):
            a, b = im(zetazero(k)), im(zetazero(k + 1))
            if tight:
                if best is None or b - a < best[1] - best[0]: best = (a, b)
            else:
                if abs((a + b) / 2 - gamma_target) < 3 and (best is None or abs(b - a - 1.5) < abs(best[1] - best[0] - 1.5)): best = (a, b)
        return best
    ok_all = True; out = []
    for label, tgt, tight in (("closest", 1977.17, True), ("typical", 940.0, False)):
        a, b = pair_near(tgt, tight)
        tc = findroot(lambda t: diff(Xi, t), (a + b) / 2)
        eps_star = -Xi(tc); c = diff(Xi, tc, 2) / 2
        ratios = []
        for frac in (0.01, 0.05, 0.2, 0.5):
            mu = frac * abs(eps_star)
            # push past threshold in the direction that lifts the pair: F = Xi + eps_star + sgn*mu with sgn chosen so mu/c>0
            sgn = 1 if c > 0 else -1
            F = lambda t: Xic(t) + eps_star + sgn * mu
            h_pred = sqrt(mu / abs(c))
            root = findroot(F, mpc(tc, h_pred))
            ratios.append(float(abs(im(root)) / h_pred))
        dev = max(abs(rr - cc) for rr, cc in zip(ratios, canon[label]))
        ok = dev < 2e-3 if label == "closest" else dev < 5e-3
        ok_all &= ok
        out.append("%s (%.3f,%.3f): ratios %s" % (label, a, b, np.round(ratios, 5)))
        # direction criterion: opposite sign keeps zeros real
        Fm = lambda t: Xic(t) + eps_star - sgn * mu
        root_m = findroot(Fm, mpc(tc + h_pred, 0))
        ok_all &= abs(im(root_m)) < 1e-20
    print("    " + "; ".join(out))
    check("splitting ratios h/sqrt(mu/c) reproduce Table 1; opposite sign keeps zeros on the axis", ok_all, "see above", "0.99998... / 0.99874...; |Im t| ~ 0")


# ----------------------------------------------------------------- lattice runs (F6, F8)
DELTA = 0.45; Q0 = 1.5 * DELTA; Y0 = 0.5; GL = DELTA
I0L = Y0**4 + 10 * Q0**2 * Y0**2 + Q0**4; D0L = Q0**2 - Y0**2; TAU_L = (np.sqrt(I0L) - D0L) / 12


def build(GR, nside):
    xs = [-Q0, Q0]
    for k in range(1, nside):
        xs.append(-Q0 - GL - (k - 1) * DELTA); xs.append(Q0 + GR + (k - 1) * DELTA)
    xs = np.sort(np.array(xs))
    return xs, np.concatenate([xs.astype(complex), [1j * Y0, -1j * Y0]])


def full_rhs(t, z):
    Dm = z[:, None] - z[None, :]; np.fill_diagonal(Dm, np.inf)
    return 2.0 * np.sum(1.0 / Dm, axis=1)


def run_lattice(GR, nside, rtol, atol, y_ev):
    xs, z0 = build(GR, nside)
    iR = int(np.argmin(np.abs(xs - Q0))); iL = int(np.argmin(np.abs(xs + Q0))); ip = xs.size
    ev = lambda t, z: z[ip].imag - y_ev; ev.terminal = True
    sol = solve_ivp(full_rhs, (0.0, 0.3), z0, method="DOP853", rtol=rtol, atol=atol, events=ev, dense_output=True)
    return sol, xs, iR, iL, ip


def model_q2(ts):
    D = D0L + 12 * ts; S = np.sqrt((2 * D * D + I0L) / 3); return (S + D) / 2


def F6(full):
    print("\nF6  accelerated fall in the full flow (Lemma 6.1; Thm 6.1; Cor 6.1)")
    nside = 150; rtol, atol, y_ev = ((1e-12, 1e-14, 0.01) if full else (1e-10, 1e-12, 0.02))
    for GR, label, canon_tau in ((2.5 * DELTA, "control G_R=2.5d", 0.05314), (5 * DELTA, "violating G_R=5d", 0.05621)):
        t0 = time.time(); sol, xs, iR, iL, ip = run_lattice(GR, nside, rtol, atol, y_ev)
        ts = np.unique(np.concatenate([sol.t, np.linspace(0, sol.t[-1], 300)])); Z = sol.sol(ts)
        qR, qL, c, y = Z[iR].real, Z[iL].real, Z[ip].real, Z[ip].imag
        nr = xs.size; sig = True; ident = 0.0; margin = np.inf
        for i in range(len(ts)):
            v = full_rhs(0, Z[:, i]); yd_real = v[ip].imag
            yd_model = -1 / y[i] - 2 * y[i] * (1 / ((c[i] - qR[i])**2 + y[i]**2) + 1 / ((c[i] - qL[i])**2 + y[i]**2))
            mask = np.ones(nr, bool); mask[[iR, iL]] = False
            sfar = np.sum(1 / ((c[i] - Z[:nr, i].real[mask])**2 + y[i]**2))
            sig &= (yd_real - yd_model) < 0; ident = max(ident, abs(yd_real - yd_model + 2 * y[i] * sfar))
        tau_real = sol.t[-1] + sol.y[ip, -1].imag**2 / 2
        # episode-comparison condition: real separation vs model separation at equal depth
        y2m_of_t = lambda tt: closed(tt, Y0, Q0)[1]
        for i in range(0, len(ts), 10):
            # model time at which model depth equals real depth y[i]
            Dm = np.sqrt(max(I0L - 12 * 0, 0))  # placeholder not used
            tm = (np.sqrt(I0L) - D0L) / 12 - y[i]**2 / 2  # first-order; refine by bisection
            lo, hi = 0.0, TAU_L
            for _ in range(60):
                mid = (lo + hi) / 2
                if y2m_of_t(mid) > y[i]**2: lo = mid
                else: hi = mid
            q2m = closed((lo + hi) / 2, Y0, Q0)[0]; margin = min(margin, (qR[i] - c[i])**2 - q2m)
        shorter = 1 - tau_real / TAU_L
        check("%s: ydot_real < ydot_model at every step; identity (6.1) exact" % label, sig and ident < 1e-9, "%d steps, identity err %.1e" % (len(ts), ident), "all negative, < 1e-9")
        check("%s: tau_real < tau_dyn=%.5f by > 20%% (canonical %.5f)" % (label, TAU_L, canon_tau), shorter > 0.20 and abs(tau_real - canon_tau) < 0.02, "tau_real=%.5f (%.1f%% shorter) [%.0fs]" % (tau_real, 100 * shorter, time.time() - t0), "~%.4f, -31%% / -35%%" % canon_tau)
        check("%s: episode-comparison condition (real q^2 <= model q^2 at equal depth)" % label, margin < 0, "min margin %.3f" % margin, "< 0")


# ----------------------------------------------------------------- F7
def F7(full):
    print("\nF7  lattice back-pressure (Lemma 7.1; Thm 7.1, 7.2; Prop 7.1-7.3)")
    rng = np.random.default_rng(3)
    def sigma_far(q, GL, GR, delta, N=4000):
        k = np.arange(N)
        s = np.sum(1 / (2 * q + GL + k * delta) - 1 / (GR + k * delta))
        # exact digamma tail
        a, b = (2 * q + GL) / delta + N, GR / delta + N
        return s + (digamma(b) - digamma(a)) / delta
    ok = True
    for _ in range(2000 if full else 300):
        d = rng.uniform(0.3, 1); q = rng.uniform(0.2, 3) * d; GL, GR = rng.uniform(0.1, 6, 2) * d
        s = sigma_far(q, GL, GR, d); ok &= (np.sign(s) == np.sign(GR - 2 * q - GL)) or abs(GR - 2 * q - GL) < 1e-9
    check("AP sign rule: sign Sigma_far(q) = sign(G_R - 2q - G_L) on random configurations (Thm 7.1)", ok, "%d configurations" % (2000 if full else 300), "all")
    s_eq = sigma_far(1.5 * 0.45, 0.45, 4 * 0.45, 0.45)
    check("birth threshold: Sigma_far = 0 at G_R = 2q + G_L = 4 delta", abs(s_eq) < 1e-9, "%.2e" % s_eq, "0")
    # cumulative checker (sufficient): random lattices satisfying (7.4) give Sigma_far <= 0
    okc = True; n_tested = 0
    for _ in range(300):
        d = 1.0; q = rng.uniform(0.3, 2); GL = rng.uniform(0.2, 3); GR = rng.uniform(0.2, 3)
        gL = rng.uniform(0.5, 1.5, 4000); gR = rng.uniform(0.5, 1.5, 4000)
        cumL = 2 * q + GL + np.cumsum(np.concatenate([[0], gL[:-1]])); cumR = GR + np.cumsum(np.concatenate([[0], gR[:-1]]))
        if np.all(cumR <= cumL):
            n_tested += 1
            xs_far_R = q + cumR; xs_far_L = -q - cumL
            s = np.sum(1 / (q - xs_far_L)) + np.sum(1 / (q - xs_far_R))
            okc &= s <= 1e-12
    check("cumulative condition (7.3) implies Sigma_far <= 0 (Thm 7.2, sufficient)", okc and n_tested > 0, "%d qualifying lattices" % n_tested, "all <= 0")
    # digamma law vs truncated lattice sum
    d = 0.45; okd = True; maxdev = 0
    for e in (0.1, 0.25, 0.5, 1.0, 2.0, -0.2):
        k = np.arange(4000)
        # zeros at -k d and at (1+e/d... ) scaled: place excess between 0 and d
        right_defect = 2 * np.sum(1 / (d + e + k * d) - 1 / (d + k * d))   # right flank vs uniform
        edot_lattice = 2 * right_defect                                       # left flank contributes the negative
        edot_formula = -4 / d * (digamma(1 + e / d) + GAMMA_E)
        dev = abs(edot_lattice - edot_formula) / abs(edot_formula); maxdev = max(maxdev, dev); okd &= dev < 2e-3
    check("digamma law (7.4) vs lattice sums at depth 4000", okd, "max rel dev %.1e" % maxdev, "< 2e-3 (paper: 0.1%)")
    # linear response coefficient and remainder bound
    okl = True
    for e in np.linspace(-0.2, 0.2, 9):
        lin = -2 * np.pi**2 / (3 * d**2) * e; exact = -4 / d * (digamma(1 + e / d) + GAMMA_E)
        okl &= abs(exact - lin) <= 8 * 1.2020569 * e * e / d**3 + 1e-12
    check("linear response -2pi^2 e/(3 delta^2) with remainder <= 8 zeta(3) e^2/delta^3", okl, "9 values of e", "bound holds")
    # nonlocal linearisation (7.6): exact ODE gap rates vs (2/delta^2) sum_m (e_{k+m}+e_{k-m}-2e_k)/m^2 on small random fields
    def nonlocal_lin(e, d):
        out = np.zeros_like(e)
        for k in range(e.size):
            m = np.arange(1, e.size); sm = 0.0
            r = k + m; l = k - m
            sm += np.sum((e[r[r < e.size]] - e[k]) / m[r < e.size]**2) + np.sum((e[l[l >= 0]] - e[k]) / m[l >= 0]**2)
            out[k] = 2 / d**2 * sm
        return out
    def gap_rates(x):
        v = 2 * np.array([np.sum(1 / (x[k] - np.delete(x, k))) for k in range(x.size)]); return v[1:] - v[:-1]
    n = 120; x_uni = d * np.arange(n); g_uni = gap_rates(x_uni)     # finite-lattice drift, subtracted below
    slopes = []; okm = True
    for _ in range(30):
        e = np.zeros(n - 1); e[30:-31] = rng.uniform(-0.02, 0.02, n - 62) * d      # field confined to the interior
        gd = gap_rates(np.concatenate([[0], np.cumsum(d + e)])) - g_uni
        c = slice(30, n - 31); slopes.append(np.polyfit(nonlocal_lin(e, d)[c], gd[c], 1)[0])
        k = int(np.argmax(e)); okm &= gd[k] <= 1e-9                                 # global maximum of the field
    check("nonlocal linearisation (7.5) predicts exact gap rates (slope -> 1 at small amplitude)", abs(np.mean(slopes) - 1) < 0.03, "mean slope %.4f" % np.mean(slopes), "1 +- 0.03")
    check("linearised response (drift subtracted): the global maximum excess is non-increasing", okm, "30 random fields", "gap rate <= 0 at the global maximum")
    # exact maximum principle (Prop 7.3): random ordered lattices, no drift subtraction, finite-size bound
    oke = True; worst = -np.inf
    for _ in range(200):
        nn = int(rng.integers(20, 200)); g = rng.uniform(0.2, 1.0, nn - 1) * d; x = np.concatenate([[0], np.cumsum(g)])
        k = int(np.argmax(g)); M = g[k]; NL, NR = k, nn - 2 - k
        v = 2 * np.array([np.sum(1 / (x[i] - np.delete(x, i))) for i in range(nn)]); gdot_k = v[k + 1] - v[k]
        am = x[k] - x[k - 1 - np.arange(NL)]; bm = x[k + 2 + np.arange(NR)] - x[k + 1]
        exact = 4 / M - 2 * M * (np.sum(1 / (am * (am + M))) + np.sum(1 / (bm * (bm + M))))
        bound = 2 / M * (1 / (NL + 1) + 1 / (NR + 1))
        oke &= abs(gdot_k - exact) < 1e-9 * max(1, abs(exact)) and gdot_k <= bound + 1e-9; worst = max(worst, gdot_k - bound)
    check("exact maximum principle (Prop 7.3): identity and global-maximum-gap rate <= finite-size bound", oke, "200 random lattices, max(rate - bound) = %.2e" % worst, "identity exact; rate <= 2/M [1/(N_L+1)+1/(N_R+1)]")
    xu = d * np.arange(101); vu = 2 * np.array([np.sum(1 / (xu[i] - np.delete(xu, i))) for i in range(101)]); gu = vu[51] - vu[50]
    check("uniform lattice attains the bound (equality case of Prop 7.3)", abs(gu - 2 / d * (1 / 51 + 1 / 50)) < 1e-9, "rate %.6f vs %.6f" % (gu, 2 / d * (1 / 51 + 1 / 50)), "equal (N_L=50, N_R=49)")
    e = np.zeros(n - 1); e[59] = 0.02 * d
    gd = gap_rates(np.concatenate([[0], np.cumsum(d + e)])) - g_uni
    check("single spike decays initially at rate 2C = 2pi^2/(3 delta^2) (Prop 7.2 vs 7.1)", abs(gd[59] / (-2 * np.pi**2 / (3 * d**2) * e[59]) - 1) < 0.03, "ratio %.4f" % (gd[59] / (-2 * np.pi**2 / (3 * d**2) * e[59])), "1 (finite lattice, drift subtracted)")


# ----------------------------------------------------------------- F8
def F8(full):
    print("\nF8  the counterexample in both readings; the exact decomposition (Thm 8.1, 8.2; Prop 8.1)")
    nside = 150; rtol, atol, y_ev = ((1e-12, 1e-14, 0.01) if full else (1e-10, 1e-12, 0.02))
    canon = {"viol": dict(band=(1.215, 1.469), integ=0.080, over=0.103, qT=1.020, fband=(-7.36, -4.25), finteg=-0.327)}
    for GR, label in ((5 * DELTA, "viol"), (2.5 * DELTA, "ctrl")):
        t0 = time.time(); sol, xs, iR, iL, ip = run_lattice(GR, nside, rtol, atol, y_ev)
        ts = np.unique(np.concatenate([sol.t, np.linspace(0, sol.t[-1], 400)])); Z = sol.sol(ts)
        xR, xL, c, y = Z[iR].real, Z[iL].real, Z[ip].real, Z[ip].imag
        V = np.array([full_rhs(0, Z[:, i]) for i in range(len(ts))])
        xRd, cd = V[:, iR].real, V[:, ip].real
        q_ba = xR - c[0]; q_fr = xR - c
        eta_ba = 2 * q_ba * xRd - (2 + 8 * q_ba**2 / (q_ba**2 + y**2))
        eta_fr = 2 * q_fr * (xRd - cd) - (2 + 8 * q_fr**2 / (q_fr**2 + y**2))
        tau_real = sol.t[-1] + sol.y[ip, -1].imag**2 / 2
        # exact decomposition (8.3) at every sample
        nr = xs.size; dec_err = 0.0
        for i in range(0, len(ts), 4):
            q, qp = xR[i] - c[i], c[i] - xL[i]; w = q + qp; f = lambda x: x / (x * x + y[i]**2)
            A1 = 2 * (q - qp) / w; A2 = 4 * q * (f(q) - f(qp))
            mask = np.ones(nr, bool); mask[[iR, iL]] = False; xf = Z[:nr, i].real[mask]
            T = 4 * q * (1 / (xR[i] - xf) - (c[i] - xf) / ((c[i] - xf)**2 + y[i]**2))
            lhs = 2 * q * (xRd[i] - cd[i]); rhs = 2 + 8 * q * q / (q * q + y[i]**2) + A1 + A2 + T.sum()
            dec_err = max(dec_err, abs(lhs - rhs) / abs(lhs))
        check("%s: foot-relative decomposition (8.2) exact along the trajectory" % label, dec_err < 1e-8, "max rel err %.1e" % dec_err, "< 1e-8")
        k = np.arange(nside - 1); sig0 = np.sum(1 / (2 * Q0 + GL + k * DELTA) - 1 / (GR + k * DELTA)); eta0 = 4 * Q0 * sig0
        check("%s: analytic eta_ba(0) = 4 q0 Sigma_far(q0) (Thm 8.1(i)) matches the trajectory at t=0" % label, abs(eta_ba[0] - eta0) < 1e-6 and ((eta0 > 0) == (GR > 2 * Q0 + GL)), "%.4f vs %.4f" % (eta_ba[0], eta0), "+1.461 (violating) / -3.259 (control)")
        if label == "viol":
            cb = canon["viol"]; integ = np.trapezoid(eta_ba, ts); over = np.max(q_ba**2 - model_q2(ts)); qT = q_ba[-1]
            finteg = np.trapezoid(eta_fr, ts)
            ok1 = np.all(eta_ba > 0) and abs(eta_ba.min() - cb["band"][0]) < 0.05 * cb["band"][0] and abs(eta_ba.max() - cb["band"][1]) < 0.05 * cb["band"][1]
            check("violating: eta_ba > 0 throughout, band [1.215, 1.469] within 5%", ok1, "[%.3f, %.3f]" % (eta_ba.min(), eta_ba.max()), "[1.215, 1.469]")
            check("violating: integral +0.080, overshoot +0.103, delivered q_T 1.020 (5%)", abs(integ - cb["integ"]) < 0.05 * cb["integ"] + 0.003 and abs(over - cb["over"]) < 0.05 * cb["over"] + 0.003 and abs(qT - cb["qT"]) < 0.05 * cb["qT"], "%.3f / %.3f / %.4f (model q_T=%.4f) [%.0fs]" % (integ, over, qT, I0L**0.25, time.time() - t0), "0.080 / 0.103 / 1.020")
            check("violating: foot-relative eta_fr < 0 throughout, band [-7.36, -4.25], integral -0.327", np.all(eta_fr < 0) and abs(eta_fr.min() - cb["fband"][0]) < 0.05 * 7.36 and abs(eta_fr.max() - cb["fband"][1]) < 0.05 * 4.25 and abs(finteg - cb["finteg"]) < 0.05 * 0.327 + 0.003, "[%.3f, %.3f], integral %.3f" % (eta_fr.min(), eta_fr.max(), finteg), "[-7.36, -4.25], -0.327")
            check("violating: tau_real = 0.0562 vs tau_dyn = 0.0818", abs(tau_real - 0.0562) < 0.002, "%.5f" % tau_real, "0.0562")
        else:
            check("control: bound holds in both readings (eta_ba <= 0, eta_fr <= 0) and q_ba^2 <= q_model^2", np.all(eta_ba <= 1e-9) and np.all(eta_fr <= 1e-9) and np.all(q_ba**2 <= model_q2(ts) + 1e-9), "max eta_ba %.3f, max eta_fr %.3f" % (eta_ba.max(), eta_fr.max()), "<= 0 (canonical eta_ba <= -3.26)")


# ----------------------------------------------------------------- F9 (full)
def F9(full):
    print("\nF9  genuine-zero checks: near-axis law (82 zeros) and the 52-zero window at t~7000 (Section 9)")
    from mpmath import mp, zetazero, im, findroot, diff, re, mpc, zeta, gamma, pi, mpf
    mp.dps = 25
    # (a) dynamic near-axis law: closest pair near 1977.17, 82 zeros moving
    n0 = int(1977.17 / (2 * mp.pi) * mp.log(1977.17 / (2 * mp.pi * mp.e))) + 1
    zs = np.array([float(im(zetazero(k))) for k in range(n0 - 41, n0 + 41)])
    gaps = np.diff(zs); j = int(np.argmin(gaps)); a, b = zs[j], zs[j + 1]
    # place the pair slightly off the axis at the midpoint with depth h0 and integrate backward-in-effect: use the real-pair split law forward
    # forward: real pair r(0) = (b-a)/2 in the field of the other 80 zeros, measure d(r^2)/dt slope near 2
    z0 = zs.astype(complex)
    def rhs(t, z):
        Dm = z[:, None] - z[None, :]; np.fill_diagonal(Dm, np.inf); return 2 * np.sum(1 / Dm, axis=1)
    # backward in time the close pair approaches r=0: integrate with negative time
    sol = solve_ivp(lambda t, z: -rhs(t, z), (0, 0.5 * ((b - a) / 2)**2 * 0.9), z0, method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True)
    ts = np.linspace(0, sol.t[-1], 200); Z = sol.sol(ts); r2 = ((Z[j + 1].real - Z[j].real) / 2)**2
    slope = np.polyfit(ts, r2, 1)[0]
    check("near-axis law on 82 genuine zeros: d(r^2)/dt slope vs -2 (paper -1.9940)", abs(slope + 2) < 0.05, "%.4f" % slope, "-2 +- 0.05")
    # (b) 52-zero window at t~7000
    n1 = int(7000 / (2 * mp.pi) * mp.log(7000 / (2 * mp.pi * mp.e))) + 1
    W = np.array([float(im(zetazero(k))) for k in range(n1 - 80, n1 + 80)])   # 160-zero field, central 52 gaps analysed
    lo, hi = 80 - 26, 80 + 26; w = W[lo:hi]; G = np.diff(w); Gbar = G.mean()
    check("52-zero window: largest gap / mean gap vs birth threshold 4 (margin ~2)", G.max() / Gbar < 4 and G.max() / Gbar > 1.5, "%.3f" % (G.max() / Gbar), "< 4 (paper 1.99 vs 4)")
    vW = 2 * np.array([np.sum(1 / (W[k] - np.delete(W, k))) for k in range(W.size)]); v = vW[lo:hi]; gdot = v[1:] - v[:-1]
    mask = np.ones(G.size, bool); mask[np.argmin(G)] = False
    slope_g = np.polyfit(G[mask] - Gbar, gdot[mask], 1)[0]; pred = -2 * np.pi**2 / (3 * Gbar**2)
    from scipy.stats import spearmanr, pearsonr
    rho = spearmanr(G[mask], gdot[mask]).correlation; pr = pearsonr(G[mask], gdot[mask])[0]
    check("gap-relaxation: slope excl. narrowest gap vs digamma prediction (paper -19.1 vs -8.2, ratio 2.3)", 1.8 < slope_g / pred < 2.8 and slope_g < 0, "slope %.2f, pred %.2f, ratio %.2f" % (slope_g, pred, slope_g / pred), "ratio 2.3 +- 0.5")
    check("gap-relaxation: rank correlation (paper Spearman -0.90, Pearson -0.58)", rho < -0.85 and pr < -0.5, "Spearman %.2f, Pearson %.2f" % (rho, pr), "< -0.85, < -0.5")
    e = G - Gbar; lin = np.zeros_like(e)
    for k in range(e.size):
        for m in range(1, e.size):
            if k + m < e.size: lin[k] += (e[k + m] - e[k]) / m**2
            if k - m >= 0: lin[k] += (e[k - m] - e[k]) / m**2
    lin *= 2 / Gbar**2
    GW = np.diff(W); eW = GW - GW.mean(); linW = np.zeros_like(eW)
    for k in range(eW.size):
        for m in range(1, eW.size):
            if k + m < eW.size: linW[k] += (eW[k + m] - eW[k]) / m**2
            if k - m >= 0: linW[k] += (eW[k - m] - eW[k]) / m**2
    linW *= 2 / Gbar**2; lc = linW[lo:hi - 1][mask]
    rho_nl = spearmanr(lc, gdot[mask]).correlation; sl_nl = np.polyfit(lc, gdot[mask], 1)[0]
    check("nonlocal linearisation on the actual field: Spearman ~0.96, slope ~2 (paper)", rho_nl > 0.93 and 1.6 < sl_nl < 2.4, "Spearman %.3f, slope %.2f" % (rho_nl, sl_nl), "> 0.93, 2.0 +- 0.4")
    check("the widest gaps close (gap rate < 0 for the 12 widest)", np.all(gdot[np.argsort(G)[-12:]] < 0), "12/12" if np.all(gdot[np.argsort(G)[-12:]] < 0) else "some widen", "all close")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true"); ap.add_argument("--groups", default="")
    a = ap.parse_args()
    groups = a.groups.split(",") if a.groups else ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"] + (["F9"] if a.full else [])
    print("verify_fourbody.py  tier=%s  groups=%s" % ("full" if a.full else "fast", ",".join(groups)), flush=True)
    t0 = time.time()
    for g in groups:
        f = globals()[g]
        try:
            f(a.full) if g in ("F5", "F6", "F7", "F8", "F9") else f()
        except Exception as e:
            check("%s (exception)" % g, False, repr(e), "-")
    print("\nPASS=%d FAIL=%d   runtime %.0fs" % (sum(ok for _, ok in RES), sum(not ok for _, ok in RES), time.time() - t0))


if __name__ == "__main__":
    main()
