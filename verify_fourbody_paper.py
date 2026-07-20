"""
verify_fourbody_paper.py -- verifier for the note
"An exactly solvable four-body model for zero collisions under the
de Bruijn heat flow" (M. Ismail).

Default run (fast, model layer):
  1. Coefficient flow: A' = -12, B' = -2A; the parabola solves it;
     I = A^2 - 12 B is t-free (symbolic, sympy).
  2. Mirror theorem: the four intertwinings of y -> i r (symbolic).
  3. Closed forms at the extremal configuration (y0, q0) = (1, 2.7684):
     I0 = 136.38, D0 = 6.664, tau_dyn = 0.41784, lifetime sqrt(I0)/6,
     crossing speed 2 sqrt(I0); the de Bruijn limit tau -> y0^2/2 as
     q0 -> infinity; the sparse floor tau >= y0^2/6 on a grid.
  4. RK4 integration of the root ODEs reproduces tau_dyn to <= 1e-6
     (measured ~3e-13) with invariant drift < 1e-10; the hyperbolic
     continuation carries J = I0 through the collision.
  5. Splitting normal form: roots of c x^2 + eps by sign of eps/c.

--deep adds (minutes; requires mpmath zetazero):
  6. Static splitting on a genuine pair of zeta zeros near gamma ~ 940:
     push Xi past the degeneracy threshold by mu = 0.01 |eps*| and check
     h_measured / sqrt(mu/c) = 0.9987(4); direction control Im ~ 1e-30s.
Requires: numpy, sympy, mpmath.
"""
import sys, math
import numpy as np

FAILED = []
def check(name, ok, extra=""):
    tag = "PASS" if ok else "FAIL"
    if not ok: FAILED.append(name)
    print(f"  [{tag}] {name}" + (f"  ({extra})" if extra else ""))

def model_layer():
    import sympy as sp
    t, A0, B0 = sp.symbols("t A_0 B_0", real=True)
    A = A0 - 12*t; B = B0 - 2*A0*t + 12*t**2
    check("A' = -12 and B' = -2A (parabola solves the flow)",
          sp.simplify(sp.diff(A, t)+12) == 0 and sp.simplify(sp.diff(B, t)+2*A) == 0)
    check("I = A^2 - 12B is conserved (t-free)",
          sp.simplify(sp.expand(A**2-12*B)-(A0**2-12*B0)) == 0)

    y, q, r = sp.symbols("y q r", positive=True)
    Ae, Be = y**2-q**2, -y**2*q**2
    Ah, Bh = -(r**2+q**2), r**2*q**2
    M = lambda e: sp.simplify(e.subs(y, sp.I*r))
    dy2 = -2-8*y**2/(y**2+q**2); dq2e = 2+8*q**2/(y**2+q**2)
    dr2 = 2-8*r**2/(q**2-r**2); dq2h = 2+8*q**2/(q**2-r**2)
    ok = (sp.simplify(M(Ae)-Ah) == 0 and sp.simplify(M(Be)-Bh) == 0
          and sp.simplify(M(dy2)+dr2) == 0 and sp.simplify(M(dq2e)-dq2h) == 0
          and sp.simplify(M(sp.expand(Ae**2-12*Be))-sp.expand(Ah**2-12*Bh)) == 0)
    check("mirror y -> i r: coefficients, vector fields, invariant", ok)

    y0, q0 = 1.0, 2.7684
    I0 = y0**4+10*q0*q0*y0*y0+q0**4; D0 = q0*q0-y0*y0
    tau = (math.sqrt(I0)-D0)/12
    check("I0 = 136.38, D0 = 6.664, tau_dyn = 0.41784",
          abs(I0-136.38) < 0.01 and abs(D0-6.664) < 5e-4 and abs(tau-0.41784) < 5e-6,
          f"tau = {tau:.6f}")
    check("lifetime sqrt(I0)/6 and crossing speed 2 sqrt(I0)",
          abs(math.sqrt(I0)/6-1.94635) < 5e-6 and abs(2*math.sqrt(I0)-23.3562) < 5e-4)
    lim = (math.sqrt(1+10*1e6+1e12)-(1e6-1))/12
    check("de Bruijn limit: tau -> y0^2/2 = 0.5 (q0 = 1000)",
          abs(lim-0.5) < 2e-6, f"{lim:.6f}")
    bad = sum(1 for yy in (0.1, 0.5, 1.0, 1.5) for qq in (1.2, 2.0, 4.0)
              if qq > yy and (math.sqrt(yy**4+10*qq*qq*yy*yy+qq**4)-(qq*qq-yy*yy))/12
              < yy*yy/6-1e-12)
    check("sparse floor tau >= y0^2/6 on grid", bad == 0)

    def rhs(s):
        yy, qq = s; d = yy*yy+qq*qq
        return np.array([-1/yy-4*yy/d, 1/qq+4*qq/d])
    s = np.array([y0, q0]); tt = 0.0; drift = 0.0
    while s[0] > 1e-8:
        h = min(1e-5, 0.002*s[0]/abs(rhs(s)[0]))
        k1 = rhs(s); k2 = rhs(s+h/2*k1); k3 = rhs(s+h/2*k2); k4 = rhs(s+h*k3)
        s = s+h/6*(k1+2*k2+2*k3+k4); tt += h
        drift = max(drift, abs(s[0]**4+10*s[1]**2*s[0]**2+s[1]**4-I0))
    check("RK4 reproduces tau_dyn to <= 1e-6; drift(I) < 1e-10",
          abs(tt-tau) < 1e-6 and drift < 1e-10,
          f"dtau {abs(tt-tau):.1e}, drift {drift:.1e}")
    qc = s[1]
    def rhs_h(s):
        rr, qq = s; d = qq*qq-rr*rr
        return np.array([1/rr-4*rr/d, 1/qq+4*qq/d])
    s2 = np.array([1e-6, qc]); 
    while s2[0] < y0:
        h = min(1e-5, 0.002*max(s2[0], 1e-5)/abs(rhs_h(s2)[0]))
        k1 = rhs_h(s2); k2 = rhs_h(s2+h/2*k1); k3 = rhs_h(s2+h/2*k2); k4 = rhs_h(s2+h*k3)
        s2 = s2+h/6*(k1+2*k2+2*k3+k4)
    J = s2[0]**4-10*s2[1]**2*s2[0]**2+s2[1]**4
    check("gluing: hyperbolic branch carries J = I0",
          abs(J-I0) < 1e-3*I0, f"|J-I0| = {abs(J-I0):.1e}")

    import cmath
    for eps_over_c, off in ((-0.04, False), (0.04, True)):
        rt = cmath.sqrt(-eps_over_c)
        check(f"normal form eps/c = {eps_over_c}: {'off-axis' if off else 'on-axis'}",
              (abs(rt.imag) > 1e-12) == off)

def deep_layer():
    import mpmath as mp
    mp.mp.dps = 30
    print("  [deep] locating a genuine pair near gamma ~ 940 (zetazero) ...")
    g1 = mp.zetazero(601).imag; g2 = mp.zetazero(602).imag
    XI = lambda tt: mp.re(0.5*(mp.mpf('0.5')+1j*tt)*((mp.mpf('0.5')+1j*tt)-1)
                          * mp.pi**(-(mp.mpf('0.5')+1j*tt)/2)
                          * mp.gamma((mp.mpf('0.5')+1j*tt)/2)
                          * mp.zeta(mp.mpf('0.5')+1j*tt))
    tc = mp.findroot(lambda u: mp.diff(XI, u), (g1+g2)/2)
    V = XI(tc); c = mp.diff(XI, tc, 2)/2
    mu = mp.mpf('0.01')*abs(V)*mp.sign(c)
    hp = mp.sqrt(mu/c)
    rfull = lambda u: (0.5*(mp.mpf('0.5')+1j*u)*((mp.mpf('0.5')+1j*u)-1)
                       * mp.pi**(-(mp.mpf('0.5')+1j*u)/2)
                       * mp.gamma((mp.mpf('0.5')+1j*u)/2)
                       * mp.zeta(mp.mpf('0.5')+1j*u))
    root = mp.findroot(lambda u: rfull(u)+(-V+mu), mp.mpc(tc, hp*mp.mpf('0.95')))
    ratio = float(abs(mp.im(root))/hp)
    check("static splitting ratio 0.9987 at mu = 0.01|eps*| (typical pair)",
          abs(ratio-0.99874) < 2e-3, f"{ratio:.5f}")
    r1 = mp.findroot(lambda u: rfull(u)+(-V-mu), tc+mp.sqrt(mu/abs(c)))
    check("direction control: opposite sign stays on axis",
          abs(mp.im(r1)) < 1e-20, f"Im = {float(abs(mp.im(r1))):.1e}")

def main():
    print("== model layer ==")
    model_layer()
    if "--deep" in sys.argv:
        print("== deep layer (genuine zeta zeros) ==")
        deep_layer()
    print()
    if FAILED:
        print("FAILURES:", ", ".join(FAILED)); raise SystemExit(1)
    print("ALL CHECKS PASS.")

if __name__ == "__main__":
    main()
