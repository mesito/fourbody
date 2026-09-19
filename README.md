# fourbody — verification suite

Companion code for

> M. Ismail, *An exactly solvable four-body model for zero collisions under the de Bruijn heat flow, and the growth of neighbour separations in the full flow* (2026).

```
fourbody/
├── paper/    Ismail_fourbody_nntdm.tex / .pdf, CCBY.png (journal template graphic)
└── code/
    ├── verify_fourbody.py   self-contained suite, groups F1–F9, fast/full tiers
    ├── runlogs/             runlog_fast.txt (PASS = 44, FAIL = 0), runlog_full_F9.txt (PASS = 6, FAIL = 0)
    └── MANIFEST.sha256
```

## Usage

```
pip install numpy scipy mpmath sympy
python3 verify_fourbody.py            # fast tier: F1–F8, ~2–3 min single core (F5 evaluates Xi with mpmath)
python3 verify_fourbody.py --full     # full tier: tighter tolerances and F9 (genuine zeros via mpmath.zetazero, ~5 min extra)
python3 verify_fourbody.py --groups F8,F6
```

The script needs no data files: the genuine zeros of ζ used in F5 and F9 are computed on the fly with `mpmath.zetazero`, and the lattice runs of F6/F8 are built from the parameters printed in the paper (300 on-line zeros, δ = 0.45, y₀ = 0.5, q₀ = 1.5δ, G_L = δ, G_R ∈ {2.5δ, 5δ}; DOP853, rtol 10⁻¹⁰ / atol 10⁻¹² in fast mode, 10⁻¹² / 10⁻¹⁴ in full mode).

Groups: F1 coefficient flow and invariant (Thm 2.1); F2 closed form, rate identity, convexity, mean rate 6, film formula (Thm 2.2, 3.1; Prop 3.1, 3.2; Cor 3.1); F3 episode structure, identity swap, reference τ_dyn = 0.4175, sparse floor, near-axis law (Thm 3.2; Cor 3.2; Prop 3.3); F4 mirror intertwinings and gluing (Thm 4.1); F5 splitting dichotomy on genuine zero pairs (Prop 5.1; Table 1); F6 accelerated fall (Lemma 6.1; Thm 6.1; Cor 6.1); F7 AP sign rule, cumulative checker, digamma law, nonlocal linear response, exact maximum principle for the global maximum gap (Lemma 7.1; Thm 7.1, 7.2; Prop 7.1–7.3); F8 the counterexample in both readings (analytic at t = 0 and numerical) and the exact decomposition (Thm 8.1, 8.2; Prop 8.1); F9 near-axis law on 82 genuine zeros and the 52-zero window at t ≈ 7000 (Section 9). Every check prints measured value, expectation with tolerance, and PASS/FAIL.
