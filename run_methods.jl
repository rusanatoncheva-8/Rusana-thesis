"
Summary:

The main purpose is to compare pulse-synthesis methods at fixed (Nmax, T) and, in particular, drive
the polynomial/structure-tensor Magnus method that is used as both propagator and 
analytic-gradient optimiser. 

Methods:
Offline precomputation of a Magnus structure tensor, unit-normalisation of the generators for
conditioning, L-BFGS on analytic trajectory gradients, with a derivative-free fallback.

Outputs:
comparison_Nmax*_T*.csv          static three-way (+ Magnus row appended)
comparison_magnus_Nmax*_T*.csv   Magnus_tensor vs GRAPE_free (bar)
magnus_pulse_Nmax*.csv           best Magnus-engine waveform
magnus_saturation_Nmax*.csv      dim(g)/d^2 + design vs honest vs buffer
magnus_convergence_Nmax*.csv     prop error vs order kM and vs nseg
mag_feasibility.csv              GRAPE self vs honest over (Nmax, buffer)
"


include(joinpath(@__DIR__, "..", "src", "RecoilFreeGate.jl"))
include(joinpath(@__DIR__, "..", "src", "MagnusMethod.jl"))
using .RecoilFreeGate
using .MagnusMethod
using LinearAlgebra, Printf, Random

const DATA = joinpath(@__DIR__, "..", "data"); mkpath(DATA)

const QUICK = true

# Operating points (Nmax, OmT) taken from the optimal time found using grape.
const POINTS  = QUICK ? [(Nmax = 1, T = 70.2)] :
                        [(Nmax = 1, T = 70.2), (Nmax = 2, T = 147.6), (Nmax = 3, T = 236.7)]

const GRAPE_RESTARTS = QUICK ? 4 : 8
const EXTRA_FIRST_ONLY = true      

# Magnus
const RUN_MAGNUS     = true
const BUFFER_DESIGN  = [4]                          # Multiple buffer designs could be tested with more time
const BUFFER_SCORE   = 4                  # Leakage-aware scoring buffer
# Convergence diagnostic 
const CONV_DEPTHS  = [2, 4, 6, 8, 10]               # Truncation orders to test
const CONV_NSEGS   = [50, 100, 200, 400, 800]       # Segment counts to test
const CONV_BUFFER  = 3                              # Small design buffer for the convergence study

# Feasibility map
const RUN_FEASIBILITY = !QUICK
const FEAS_NMAXS      = [1, 2, 3]
const FEAS_BUFFERS    = [2, 4, 6, 8, 10, 12]

#  Static three-way comparison 
"Build a comparison row from a pulse"
function row_of(method, sys, delta, T, npar, secs)
    m = metrics(sys, propagate(sys, delta, T))
    (; method, nparams = npar, nseg = length(delta),
       infid_avg = 1 - m.Favg, infid_pro = 1 - m.Fpro, Rgrade = m.Rgrade,
       leak = m.leak, peakD = maximum(abs, delta), secs)
end

"
At each row it builds the large-buffer scoring system, runs free GRAPE for the reference
row, and at the first (full) point only, when RUN_MAGNUS, it appends the Magnus row 
via magnus_row. 
It prints the table and writes comparison_Nmax*_T*.csv and returns the first (sys, T) as
the diagnostics anchor. (Later points report GRAPE_free only.)
"
function section_comparison()
    println("Method comparison")
    syss_score = nothing
    for (idx, (Nmax, T)) in enumerate(POINTS)
        sys = build_system(ETA_SR88, Nmax; omega = OMEGA_REL, buffer = BUFFER_SCORE)
        full = !EXTRA_FIRST_ONLY || idx == 1     # full comparison only at the first point
        @printf("comparison @ Nmax=%d, OmT=%.1f, B=%d, Mh=%d%s",
                Nmax, T, BUFFER_SCORE, MH, full ? "" : "  [GRAPE-only]")
        println("    running GRAPE_free ...")
        tg = @elapsed (dG = synthesize(sys, T, seg_count(T); restarts = GRAPE_RESTARTS, itc = 300, itf = 700, objective = :point))
        rows = NamedTuple[row_of("GRAPE_free", sys, dG, T, seg_count(T), tg)]
        if full
            # all three static-frame methods (reuses the proven core routine)
            #rows = compare_methods(sys, T; Mh = MH, grape_restarts = GRAPE_RESTARTS, objective = :point,include_magnus = false)
            if RUN_MAGNUS
                magrow = magnus_row(Nmax, T, sys)
                magrow === nothing || push!(rows, magrow)
            end
        end
        @printf("  %-16s %-7s %-7s %-11s %-11s %-11s %-11s %-9s %-8s\n",
                "method", "npar", "nseg", "1-Favg", "1-Fpro", "Rgrade", "leak", "peak|D|", "secs")
        for r in rows
            @printf("  %-16s %-7d %-7d %-11.3e %-11.3e %-11.3e %-11.3e %-9.3g %-8.1f\n",
                    r.method, r.nparams, r.nseg, r.infid_avg, r.infid_pro, r.Rgrade, r.leak, r.peakD, r.secs)
        end
        write_comparison_csv(joinpath(DATA, "comparison_Nmax$(Nmax)_T$(round(Int,T)).csv"), rows)
        syss_score === nothing && (syss_score = (sys, T))
    end
    syss_score
end

"""
For each design buffer it computes k_M (depth_for_design), prints dsim and k_M,
precomputes the tensor, prints the saturation fraction dim(g)/dsim², 
runs magnus_synthesize_grad, and re-scores the physical pulse on syss 
(the large BUFFER_SCORE system) so leakage hidden by the small design buffer surfaces.
It writes the saturation/scalability table, the best Magnus waveform, and a standalone
Magnus-vs-free-GRAPE comparison CSV, and returns the headline row. 
"""
function magnus_row(Nmax, T, syss)
    results = NamedTuple[]
    for b in BUFFER_DESIGN
        kM = depth_for_design(Nmax, b)
        @printf("  [Magnus] design buffer=%d  ->  dsim=%d, k_M=%d  (from {A,B} completion data)\n",
                b, 2 * (Nmax + b + 1), kM)
        pc = precompute(Nmax, b, kM, OPTOL); pc === nothing && continue
        frac = pc.dim / pc.sysd.dsim^2
        @printf("    dim(g)=%d of d^2=%d (%.0f%% of u(d); %s)\n",
                pc.dim, pc.sysd.dsim^2, 100 * frac,
                frac > 0.9 ? "saturated -- representative" : "NOT saturated at k_M=$kM")
        syn = magnus_synthesize_grad(pc, T)   
        dscore = syn.delta                                    # physical Delta(t) waveform
        ms = metrics(syss, propagate(syss, dscore, T))
        @printf("    design 1-Favg=%.3e | HONEST 1-Favg=%.3e (Rgrade=%.2e, leak=%.2e, peak|D|=%.2f, prop_err=%.1e)\n",
                syn.design_infid, 1 - ms.Favg, ms.Rgrade, ms.leak, maximum(abs, dscore), syn.prop_err)
        push!(results, (; buffer = b, dim_g = pc.dim, dsim = pc.sysd.dsim, nseg = syn.nseg,
                          nparams = syn.nparams, design_infid = syn.design_infid, honest = ms, delta = dscore))
    end
    isempty(results) && (@warn "Magnus: no design buffer was affordable (dim cap $PRECOMP_DIM_CAP)"; return nothing)

    best = results[argmin([1 - r.honest.Favg for r in results])]   # most faithful

    # saturation / scalability table
    write_csv(joinpath(DATA, "magnus_saturation_Nmax$(Nmax).csv"),
              ["buffer", "dsim", "dim_g", "frac_u_d", "nseg", "design_infid", "honest_infid", "leak"],
              Any[Float64[r.buffer for r in results], Float64[r.dsim for r in results],
                  Float64[r.dim_g for r in results], Float64[r.dim_g / r.dsim^2 for r in results],
                  Float64[r.nseg for r in results], Float64[r.design_infid for r in results],
                  Float64[1 - r.honest.Favg for r in results], Float64[r.honest.leak for r in results]])
    # best Magnus pulse waveform (one sample per design segment)
    write_csv(joinpath(DATA, "magnus_pulse_Nmax$(Nmax).csv"),
              ["t", "delta"], Any[collect(0:length(best.delta)-1) .* (T / length(best.delta)), best.delta])

    # standalone Magnus-vs-GRAPE comparison CSV (free GRAPE on the large space)
    t = @elapsed (dG = synthesize(syss, T, seg_count(T); restarts = GRAPE_RESTARTS,
                                  itc = 300, itf = 700, objective = :point))
    mg = metrics(syss, propagate(syss, dG, T))
    cmp_rows = NamedTuple[
        (method = "Magnus_tensor", nparams = best.nparams, nseg = best.nseg,
         infid_avg = 1 - best.honest.Favg, infid_pro = 1 - best.honest.Fpro,
         Rgrade = best.honest.Rgrade, leak = best.honest.leak,
         peakD = maximum(abs, best.delta), secs = 0.0),
        (method = "GRAPE_free", nparams = seg_count(T), nseg = seg_count(T),
         infid_avg = 1 - mg.Favg, infid_pro = 1 - mg.Fpro, Rgrade = mg.Rgrade,
         leak = mg.leak, peakD = maximum(abs, dG), secs = t)]
    write_comparison_csv(joinpath(DATA, "comparison_magnus_Nmax$(Nmax)_T$(round(Int,T)).csv"), cmp_rows)

    
    (method = "Magnus_tensor", nparams = best.nparams, nseg = best.nseg,
     infid_avg = 1 - best.honest.Favg, infid_pro = 1 - best.honest.Fpro,
     Rgrade = best.honest.Rgrade, leak = best.honest.leak,
     peakD = maximum(abs, best.delta), secs = 0.0)
end

" 
Part (a) is a sweep truncation order k_M ∈ CONV_DEPTHS at a fixed, well-resolved
nseg and record ‖U_magnus − U_exact‖ 
Part (b) is fixed k_M at the synthesis depth and sweep nseg as multiples
of the convergence-bound minimum ⌈αT⌉ (ŝ = αT/nseg < 1), showing the segment-count convergence.
It writes the two magnus_convergence* CSVs.
"
function section_magnus_convergence(Nmax, T)
    RUN_MAGNUS || return
    println("MagnusTensor convergence")
    sysd0 = build_system(ETA_SR88, Nmax; omega = OMEGA_REL, buffer = CONV_BUFFER)
    theta = 0.5 .* randn(MersenneTwister(20235021), 1 + 2 * MH)

    # error vs truncation order kM at a fixed, well-resolved nseg
    normH  = opnorm(Matrix(sysd0.A)) + 2.0 * opnorm(Matrix(sysd0.B))
    nseg0  = max(8, ceil(Int, T / (CONV_SAFETY * pi / normH)))
    Ue     = propagate(sysd0, sample_delta(theta, T, 8nseg0, sysd0.omega, MH), T)  # near-exact static
    order_err = Float64[]
    for kM in CONV_DEPTHS
        pc = precompute(Nmax, CONV_BUFFER, kM, OPTOL)
        if pc === nothing; push!(order_err, NaN); continue; end
        Um = magnus_propagate(pc.dyn, pc.Lbasis, theta, T, nseg0, MDEG, sysd0.omega, MH, pc.alpha, pc.beta)
        push!(order_err, Um === nothing ? NaN : opnorm(Um - Ue))
        @printf("  order kM=%-2d nseg=%d : ||U_magnus - U_exact|| = %.3e\n", kM, nseg0, order_err[end])
    end
    write_csv(joinpath(DATA, "magnus_convergence_order_Nmax$(Nmax).csv"),
              ["kM", "nseg", "err"], Any[Float64.(CONV_DEPTHS), fill(Float64(nseg0), length(CONV_DEPTHS)), order_err])

    # error vs segment count at the synthesis order k_M (from the {A,B} data)
    synth_depth = depth_for_design(Nmax, CONV_BUFFER)
    pcD = precompute(Nmax, CONV_BUFFER, synth_depth, OPTOL)
    if pcD !== nothing
        nseg_min = ceil(Int, pcD.alpha * T)              # ŝ = α·τ < 1  ⇔  nseg > α·T
        nsegs    = [round(Int, pcD.alpha * T * f) for f in (1.2, 1.5, 2.0, 3.0, 4.0)]
        seg_err  = Float64[]
        for ns in nsegs
            shat = pcD.alpha * T / ns
            Um = magnus_propagate(pcD.dyn, pcD.Lbasis, theta, T, ns, MDEG, sysd0.omega, MH, pcD.alpha, pcD.beta)
            push!(seg_err, Um === nothing ? NaN : opnorm(Um - Ue))
            @printf("  depth=%d  nseg=%-5d ŝ=%.2f : ||U_magnus - U_exact|| = %.3e\n",
                    synth_depth, ns, shat, seg_err[end])
        end
        write_csv(joinpath(DATA, "magnus_convergence_Nmax$(Nmax).csv"),
                  ["nseg", "err"], Any[Float64.(nsegs), seg_err])
    end
    println("  wrote magnus_convergence_order_Nmax$(Nmax).csv and magnus_convergence_Nmax$(Nmax).csv")
end


"
For each (Nmax, buffer) it designs and scores GRAPE at buffer b, then re-scores honestly at
BUFFER_SCORE, exposing the overfitting gap between design-buffer and honest fidelity. 
Writes mag_feasibility.csv.
"
function section_feasibility()
    RUN_FEASIBILITY || return
    println("Feasibility map (self vs honest)")
    path = joinpath(DATA, "mag_feasibility.csv")
    open(path, "w") do io; println(io, "Nmax,buffer,dsim,T,self_infid,honest_infid"); end
    for Nmax in FEAS_NMAXS
        T  = 50.0 + 25.0 * Nmax
        sb = BUFFER_SCORE
        syss = build_system(ETA_SR88, Nmax; omega = OMEGA_REL, buffer = sb)
        @printf("  Nmax=%d (T=%.0f, score buffer=%d):\n", Nmax, T, sb)
        for b in FEAS_BUFFERS
            b > sb && continue
            sysb = build_system(ETA_SR88, Nmax; omega = OMEGA_REL, buffer = b)
            dd   = synthesize(sysb, T, seg_count(T); restarts = 3, itc = 200, itf = 400, objective = :point)
            self = 1 - metrics(sysb, propagate(sysb, dd, T)).Favg                # designed & scored at b
            hon  = b == sb ? self : 1 - metrics(syss, propagate(syss, dd, T)).Favg # honest re-score
            @printf("    buffer=%-3d dsim=%-3d self=%.3e honest=%.3e\n", b, sysb.dsim, self, hon)
            open(path, "a") do io
                println(io, @sprintf("%d,%d,%d,%.1f,%.6e,%.6e", Nmax, b, sysb.dsim, T, self, hon))
            end
        end
    end
    println("  wrote ", basename(path))
end

function main()
    println("Method comparison incl. MagnusTensor (Raul, arXiv:2512.20357)")
    println("eta=$(ETA_SR88), omega/Omega=$(OMEGA_REL), score buffer=$(BUFFER_SCORE), QUICK=$(QUICK)")
    HAVE_OPTIM || @warn "Optim.jl not found; L-BFGS unavailable (Adam fallback is slower)."

    diag = section_comparison()                       # Section 1 (+ Magnus rows)
    if diag !== nothing
        sys0, T0 = diag
        section_magnus_convergence(sys0.Nmax, T0)     # Section 3
    end
    section_feasibility()                             # Section 4

    println("\nDone. CSVs in ", DATA, " ; render with scripts/visualize.jl")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
