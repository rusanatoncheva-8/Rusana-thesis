#  run_studies2.jl

"
Summary:
This is the main script where the results for the paper will be generated.
The script consists of four studies, all using free-segment GRAPE, the approach that was found to beoptimal:
Study A is the minimum-time / fidelity-floor scaling vs Nmax;
Study B is the thermal figure of merit at a register matched to a thermal n̄; 
Study C (phase4) is time-resolved diagnostics of a few optimal pulses;
And phase space the conditional motional (X,P) trajectory with a loop-closure metric.

Methods:
All synthesis is free-segment GRAPE via synthesize/opt_gate from the core module.

Output:
studyA_curve_Nmax*.csv, studyA_summary.csv, studyA_scaling.csv,
studyB.csv, pulse_Nmax*_<obj>.csv, phasespace_Nmax*.csv, phasespace_closure.csv.
"


include(joinpath(@__DIR__, "..", "src", "RecoilFreeGate.jl"))
using .RecoilFreeGate, Printf

const DATA = joinpath(@__DIR__, "..", "data"); mkpath(DATA)

const RUN_STUDY_A = true
const RUN_STUDY_B = true
const RUN_PHASE4  = true
const RUN_PHASESPACE = true         # export motional (X,P) trajectories of the best pulse per Nmax
const A_NMAXS     = 1:6
const B_NMAXS     = [2, 6]          # the bigger, the more expensive
const PS_NMAXS    = 1:6             # phase-space
const NBAR        = 0.5


const REPORT_THRESHOLDS = [1e-3, 1e-4, 1e-5]   # crossings recorded
const SCALING_THRESHOLD = 1e-4                 # OmT* fit at this level 

"
The per-Nmax optimiser floor target (5e-6 for Nmax≤3, else 1e-5); 
the achievable floor degrades with Nmax in the leakage-limited regime.
"
target_infid(Nmax) = Nmax <= 3 ? 5e-6 : 1e-5


"The smallest gate time whose floor infidelity is ≤ thr, by log-linear interpolation on
the sorted OmT-infidelity curve: T* = T_{k-1} + (ln thr - ln i_{k-1})/(ln i_k - ln i_{k-1})·(T_k - T_{k-1}).
Returns NaN if never crossed. Log-linear because infidelity falls roughly exponentially
with duration over the relevant range."
function cross_at(Ts, infids, thr)
    isempty(Ts) && return NaN
    p = sortperm(Ts); Tss = Ts[p]; ii = infids[p]
    for k in eachindex(Tss)
        if ii[k] <= thr
            k == 1 && return Tss[1]
            y1 = log(ii[k-1]); y2 = log(ii[k])
            (!isfinite(y1) || !isfinite(y2) || abs(y2 - y1) < eps()) && return Tss[k]
            return Tss[k-1] + (log(thr) - y1) / (y2 - y1) * (Tss[k] - Tss[k-1])
        end
    end
    NaN
end
# <<<

# Study A
# Aims to find the absolute shortest pulse duration that can still achieve an accurate gate

"
One optimisation at fixed (sys, T): assembles a warm-started candidate (opt_gate from
the upsampled seed, when a seed is given) and a fresh synthesize (when restarts > 0), scores
each by Favg, and returns the best with 1-Favg, leak, and S = seg_count(T). Taking the better of
warm-start and fresh makes the descent robust without paying a full multistart every step.
"
function _solve(sys, T, seed; restarts, itc, itf)
    S = seg_count(T)
    cands = Vector{Vector{Float64}}()
    seed === nothing || push!(cands, opt_gate(sys, upsample(seed, S), T, itf; objective = :point))
    restarts > 0 && push!(cands, synthesize(sys, T, S; restarts, itc, itf, objective = :point))
    isempty(cands) && push!(cands, synthesize(sys, T, S; restarts = 1, itc, itf, objective = :point))
    ms = [metrics(sys, propagate(sys, c, T)) for c in cands]
    i  = argmax([m.Favg for m in ms])
    (; d = cands[i], inf = 1 - ms[i].Favg, leak = ms[i].leak, S)
end

"""
Find the shortest gate time which fidelity floor meets the target fidelity, for one system.
Returns (Nmax, Ts, infids, leaks, Tstar, floorI, floorT) with the full curve.

It  starts with a longer gate time Tanchor, where finding a good pulse should be easy.
Then it increases the time if needed (if the target gate fidelity was not achieved).
Once a successful pulse was found, the script reduces the gate time by 10% and it uses
the optimal shape from the last pulse as a warm start.
Keeps srinking the time and re-optimising until the pulse fails.
Then bisects the difference between the last passing time and the failing time to pinpoint
the exact minimum boundary.
"""
function min_time_down(sys; target, restarts = 4, itc = 300, itf = 800,
                       Tanchor = nothing, shrink = 0.90, max_down = 14, max_up = 5,
                       growth = 1.22, plateau = 0.8)
    Nmax = sys.Nmax
    Ts = Float64[]; infids = Float64[]; leaks = Float64[]
    function log!(tag, T, r)
        push!(Ts, T); push!(infids, r.inf); push!(leaks, r.leak)
        @printf("    [%-6s] OmT=%-7.1f S=%-4d 1-Favg=%-10.2e leak=%-10.2e\n", tag, T, r.S, r.inf, r.leak)
    end

    # Anchor
    T = isnothing(Tanchor) ? round(45.0 + 38.0 * Nmax, digits = 1) : Tanchor
    r = _solve(sys, T, nothing; restarts, itc, itf); log!("anchor", T, r)
    seed = r.d; bestinf = r.inf; stall = 0; up = 0
    while r.inf > target && up < max_up
        T = round(T * growth, digits = 1); up += 1
        r = _solve(sys, T, seed; restarts = max(2, restarts - 2), itc, itf); log!("grow", T, r); seed = r.d
        if r.inf < plateau * bestinf
            bestinf = r.inf; stall = 0
        else
            stall += 1
        end
        stall >= 2 && (@printf("    (floor plateaued; target not reached)\n"); break)
    end

    if r.inf > target                      # target below the achievable floor
        fi = argmin(infids)
        @printf("    -> target %.1e NOT reached; best 1-Favg=%.2e @ OmT=%.1f\n", target, infids[fi], Ts[fi])
        return (; Nmax, Ts, infids, leaks, Tstar = NaN, floorI = infids[fi], floorT = Ts[fi])
    end

    # Push down
    Tstar = T; Tpass = T; down = 0
    while down < max_down
        Tn = round(Tpass * shrink, digits = 1); down += 1
        r = _solve(sys, Tn, seed; restarts = 0, itc, itf); log!("down", Tn, r)   # warm start only
        if r.inf > target
            r = _solve(sys, Tn, seed; restarts = max(2, restarts - 2), itc, itf); log!("retry", Tn, r)
        end
        if r.inf > target
            Tm = round(0.5 * (Tn + Tpass), digits = 1)                            # bisect the boundary once
            rb = _solve(sys, Tm, seed; restarts = max(2, restarts - 2), itc, itf); log!("bisect", Tm, rb)
            rb.inf <= target && (Tstar = Tm)
            break
        end
        Tstar = Tn; Tpass = Tn; seed = r.d                                        # success - keep descending
    end

    fi = argmin(infids)
    (; Nmax, Ts, infids, leaks, Tstar, floorI = infids[fi], floorT = Ts[fi])
end

"
Runs min_time_down per Nmax ∈ A_NMAXS at its target_infid on DEMO_BUFFER; writes the
full curve; computes the fixed-threshold crossings cr3/cr4/cr5 and then writes studyA_summary.csv; 
The script aslo fits the scaling on the 1e-4 crossing, both a linear
OmT* ≈ a + b·Nmax and a power law OmT* ∼ Nmax^b via _linfit to studyA_scaling.csv.
It is reporting both fits and lets the data choose the form.
"
function study_A()
    println("Fidelity floor & shortes-pulse search vs Nmax (B=$(DEMO_BUFFER))")
    rows = NamedTuple[]
    for Nmax in A_NMAXS
        tgt = target_infid(Nmax)
        @printf("Nmax = %d (target %.1e)\n", Nmax, tgt)
        sys = build_system(ETA_SR88, Nmax; omega = OMEGA_REL, buffer = DEMO_BUFFER)
        sc  = min_time_down(sys; target = tgt, restarts = 4, itc = 300, itf = 800)
        write_csv(joinpath(DATA, "studyA_curve_Nmax$(Nmax).csv"),
                  ["OmT", "infid_avg", "leak"], Any[sc.Ts, sc.infids, sc.leaks])

        cr = Dict(thr => cross_at(sc.Ts, sc.infids, thr) for thr in REPORT_THRESHOLDS)
        @printf("  Nmax=%d : OmT*(strict %.0e)=%s | OmT*(1e-4)=%s ; floor 1-Favg=%.2e @ OmT=%.1f\n",
                Nmax, tgt, isnan(sc.Tstar) ? "-" : @sprintf("%.1f", sc.Tstar),
                isnan(cr[1e-4]) ? "-" : @sprintf("%.1f", cr[1e-4]), sc.floorI, sc.floorT)
        push!(rows, (; Nmax, Tstar = sc.Tstar, floorI = sc.floorI, floorT = sc.floorT,
                       target = tgt, cr3 = cr[1e-3], cr4 = cr[1e-4], cr5 = cr[1e-5]))
        # <<<
    end

    open(joinpath(DATA, "studyA_summary.csv"), "w") do io
        println(io, "Nmax,target_infid,OmTstar_strict,floor_infid_avg,T_floor,OmT_1e3,OmT_1e4,OmT_1e5")
        for r in rows
            println(io, @sprintf("%d,%.1e,%.6g,%.6e,%.6g,%.6g,%.6g,%.6g",
                    r.Nmax, r.target, r.Tstar, r.floorI, r.floorT, r.cr3, r.cr4, r.cr5))
        end
    end

    Ns = Float64[r.Nmax for r in rows]
    y4 = Float64[r.cr4 for r in rows]
    ok = isfinite.(y4)
    if count(ok) >= 2
        lin = _linfit(Ns[ok], y4[ok])
        pw  = _linfit(log.(Ns[ok]), log.(y4[ok]))
        @printf("\n  scaling OmT*(1e-4): linear %.2f + %.2f*Nmax | power ~ Nmax^%.2f  (n=%d)\n",
                lin.a, lin.b, pw.b, count(ok))
        open(joinpath(DATA, "studyA_scaling.csv"), "w") do io
            println(io, "threshold,intercept_a,slope_b,power_exponent,n_points")
            println(io, @sprintf("%.1e,%.6g,%.6g,%.6g,%d", SCALING_THRESHOLD, lin.a, lin.b, pw.b, count(ok)))
        end
        println("  wrote studyA_scaling.csv")
    else
        println("\n  scaling OmT*(1e-4): too few registers crossed the threshold for a fit.")
    end
    println("studyA_summary.csv and studyA_curve_Nmax*.csv")
end


"""
Evaluates real-world performance of the synthesised pulses in a thermal motional
state. Sizes the register to capture 99%-99.9% of nbar, synthesises an optimal
pulse, and reports thermal fidelity and operational heating.
"""
function study_B()
    println("\nThermal figure of merit at register-matched Nmax (nbar=$(NBAR))")
    @printf("  nbar=%.2f needs Nmax>=%d (99%%), >=%d (99.9%%).\n",
            NBAR, recommend_nmax(NBAR; coverage = 0.99), recommend_nmax(NBAR; coverage = 0.999))
    open(joinpath(DATA, "studyB.csv"), "w") do io
        println(io, "Nmax,OmT,infid_avg,infid_thermal,dnbar_reg,Pret,outside_pop")
        for Nmax in B_NMAXS
            @printf("  --- Nmax = %d ---\n", Nmax)
            sys = build_system(ETA_SR88, Nmax; omega = OMEGA_REL, buffer = DEMO_BUFFER)
            T = round(50.0 + 30.0 * Nmax, digits = 1); S = seg_count(T)
            d = synthesize(sys, T, S; restarts = 4, itc = 300, itf = 700, objective = :point)
            U = propagate(sys, d, T); m = metrics(sys, U)
            Fth = thermal_gate_fidelity(sys, U; nbar = NBAR); h = thermal_heating(sys, U; nbar = NBAR)
            @printf("  Nmax=%d OmT=%.1f: 1-Favg=%.2e 1-Fth=%.2e dnbar=%.2e Pret=%.6f outside=%.2e\n",
                    Nmax, T, 1 - m.Favg, 1 - Fth, h.dnbar, h.Pret, h.outside_reg_pop)
            println(io, @sprintf("%d,%.1f,%.6e,%.6e,%.6e,%.8f,%.6e",
                    Nmax, T, 1 - m.Favg, 1 - Fth, h.dnbar, h.Pret, h.outside_reg_pop))
        end
    end
    println("studyB.csv")
end

# Study C
"""
Time-resolved diagnostics for a few successful pulses: Nmax=1 (point and relaxed
manifold objectives) and Nmax=2 (point).
"""
function phase4()
    println("\nOptimal-pulse analysis")
    for (Nmax, T, obj) in [(1, 70.0, :point), (1, 70.0, :manifold), (2, 100.0, :point)]
        @printf("  --- Nmax = %d (%s) ---\n", Nmax, obj)
        sys = build_system(ETA_SR88, Nmax; omega = OMEGA_REL, buffer = DEMO_BUFFER)
        d = synthesize(sys, T, seg_count(T); restarts = 4, itc = 300, itf = 700, objective = obj)
        m = metrics(sys, propagate(sys, d, T))
        path = joinpath(DATA, "pulse_Nmax$(Nmax)_$(obj).csv")
        export_pulse_csv(sys, d, T, path)
        @printf("  Nmax=%d %-9s OmT=%.0f: 1-Favg=%.2e leak=%.2e -> %s\n",
                Nmax, obj, T, 1 - m.Favg, m.leak, basename(path))
    end
    println("  wrote pulse_*.csv")
end

"
For the best pulse per Nmax, tracks the conditional motional expectations in the quadratures
X = a + a†, P = i(a† − a) along the trajectory of each motional basis input |atom=0, n⟩ (n∈{0,1}),
and reports the closure metric c_n = √((X_end−X_start)² + (P_end−P_start)²).
A recoil-free gate should return each conditional motional state to its origin, so each (X,P) loop must close
geometric, basis-independent confirmation of recoil-freeness, complementary to Rgrade and the heating.
It then writes phasespace_Nmax*.csv and phasespace_closure.csv.
"
function phase_space()
    println("\nPhase space motional (X,P) trajectory of the best pulse per Nmax")
    # >>> CHANGED: collect the closure metric and save it (was printed only).
    closure_rows = NamedTuple[]
    # <<<
    for Nmax in PS_NMAXS
        @printf("  Nmax = %d", Nmax)
        sys = build_system(ETA_SR88, Nmax; omega = OMEGA_REL, buffer = DEMO_BUFFER)
        T   = round(50.0 + 25.0 * Nmax, digits = 1)
        d   = synthesize(sys, T, seg_count(T); restarts = 3, itc = 200, itf = 400, objective = :point)
        m   = metrics(sys, propagate(sys, d, T))
        mm = sys.dmot
        a  = zeros(ComplexF64, mm, mm); for n in 1:mm-1; a[n, n+1] = sqrt(n); end
        X  = kron(sys.I2, a + a'); P = kron(sys.I2, im * (a' - a))
        rec = propagate_record(sys, d, T)
        bstate(n) = (en = zeros(ComplexF64, mm); en[n+1] = 1; kron(ComplexF64[1, 0], en))
        traj(psi0) = begin
            xs = Float64[]; ps = Float64[]
            for U in rec.Ulist
                psi = U * psi0; push!(xs, real(psi' * X * psi)); push!(ps, real(psi' * P * psi))
            end
            xs, ps
        end
        x0, p0 = traj(bstate(0)); x1, p1 = traj(bstate(1))
        c0 = hypot(x0[end] - x0[1], p0[end] - p0[1]); c1 = hypot(x1[end] - x1[1], p1[end] - p1[1])
        write_csv(joinpath(DATA, "phasespace_Nmax$(Nmax).csv"),
                  ["t", "X_n0", "P_n0", "X_n1", "P_n1"], Any[rec.ts, x0, p0, x1, p1])
        @printf("  Nmax=%d OmT=%.0f: 1-Favg=%.2e  closure|0,0>=%.2e |0,1>=%.2e -> phasespace_Nmax%d.csv\n",
                Nmax, T, 1 - m.Favg, c0, c1, Nmax)
        push!(closure_rows, (; Nmax, OmT = T, infid = 1 - m.Favg, c0, c1))
    end
    open(joinpath(DATA, "phasespace_closure.csv"), "w") do io
        println(io, "Nmax,OmT,infid_avg,closure_n0,closure_n1")
        for r in closure_rows
            println(io, @sprintf("%d,%.1f,%.6e,%.6e,%.6e", r.Nmax, r.OmT, r.infid, r.c0, r.c1))
        end
    end
    println("  wrote phasespace_Nmax*.csv and phasespace_closure.csv")
end

function main()
    println("Validation/results studies. eta=$(ETA_SR88), omega/Omega=$(OMEGA_REL), B=$(DEMO_BUFFER)")
    RUN_STUDY_A && study_A()
    RUN_STUDY_B && study_B()
    RUN_PHASE4  && phase4()
    RUN_PHASESPACE && phase_space()
    println("\nDone. CSVs in ", DATA)
end

main()