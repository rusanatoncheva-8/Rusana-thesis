#  MagnusMethod.jl
module MagnusMethod
"
Summary:

The main purpose of this module is to provide the polynomial/structure-tensor Magnus method
of dos Santos (arXiv:2512.20357), used as both a propagator and an analytic-gradient optimiser.

Methods:
Offline precomputation of a Magnus structure tensor (cached on disk); unit-normalisation of the
generators for conditioning; L-BFGS on analytic trajectory gradients, with a derivative-free fallback.

Output:
In-memory objects (precomputed tensor, synthesised pulses) and a cached structure tensor written to
data/magtensor_*.jld2.
"

using ..RecoilFreeGate
using MagnusTensor
import MagnusTensor.CalculateMagnusTensor: calculate_tensor_const_degree
import MagnusTensor.DynamicsUtils: load_dynamics_cdeg, contract_dynamics!, contract_dynamics,
                                   Trajectory, cost_trajectory, grad_cost_trajectory
using MagnusTensor.LieAlgebraUtils
using JLD2, LinearAlgebra, SparseArrays, Printf, Random
import Optim

const DATA = joinpath(@__DIR__, "..", "data"); mkpath(DATA)   # tensor cache directory

export ab_completion_depth, depth_for_design
export delta_of, fourier_taylor, sample_delta
export magnus_propagate, precompute, magnus_nseg, magnus_U, render_delta, hat_ctrl, phys_ctrl
export magnus_synthesize_grad, magnus_synthesize_nm, magnus_synthesize
export MDEG, CONV_SAFETY, OPT_RESTARTS, OPTOL, PRECOMP_DIM_CAP, MAGNUS_GRADIENT, MH, AB_COMPLETION_DEPTH

const MH             = 6           # Fourier harmonics for the comparison 
#  Magnus truncation order k_M  is set based on the obtained results from the reachability script.
const AB_COMPLETION_DEPTH = Dict(6=>8, 8=>10, 10=>10, 12=>10, 14=>12, 16=>14, 18=>14)

"k_M for a design space of register dimension dsim, from the {A,B} reachability
 data above.  Outside the measured range it extrapolates +2 per 4 extra register
 levels and rounds up to an even order (Magnus time-symmetry)."
function ab_completion_depth(dsim::Int)
    haskey(AB_COMPLETION_DEPTH, dsim) && return AB_COMPLETION_DEPTH[dsim]
    k = dsim <= 6  ? 8 :
        dsim >= 18 ? 14 + 2 * cld(dsim - 18, 4) : 10
    iseven(k) ? k : k + 1
end

"k_M for a design (Nmax, buffer):  dsim = 2(Nmax+buffer+1)."
depth_for_design(Nmax::Int, buffer::Int) = ab_completion_depth(2 * (Nmax + buffer + 1))

const MDEG           = 2           # local control polynomial degree (value+slope; stable)
const CONV_SAFETY    = 0.30        # segment tau = CONV_SAFETY * pi/||H|| (convergence bound)
const OPT_RESTARTS   = 2
const OPTOL          = 1e-6       # algebra acceptance tolerance for the tensor
const PRECOMP_DIM_CAP = 600        # skip precompute if dim(g) would exceed this
const MAGNUS_GRADIENT = true       

#  Fourier pulse + local Taylor coefficients 
"Global detuning d(t) = a0 + sum_m a_m cos(m w t) + b_m sin(m w t); theta=[a0,a1,b1,...]."
delta_of(theta, t, omega, Mh) =
    theta[1] + sum(theta[2m] * cos(m * omega * t) + theta[2m+1] * sin(m * omega * t) for m in 1:Mh)

"Local Taylor coefficients [d(ts), d'(ts), ...] of the Fourier pulse at ts (length m).
 Only value and slope are used (m=2)."
function fourier_taylor(theta, ts, m, omega, Mh)
    c = zeros(Float64, m)
    for j in 0:m-1
        s = (j == 0) ? theta[1] : 0.0
        for mm in 1:Mh
            w = mm * omega
            s += w^j * (theta[2mm] * cos(w * ts + j * pi / 2) + theta[2mm+1] * sin(w * ts + j * pi / 2))
        end
        c[j+1] = s
    end
    c
end

"Sample the smooth pulse to S piecewise-constant segments."
sample_delta(theta, T, S, omega, Mh) = [delta_of(theta, (j - 0.5) * T / S, omega, Mh) for j in 1:S]

"
Builds U(T) = ∏_s exp(-i M_s) from a Fourier control: per segment it forms the local
Taylor coefficients (fourier_taylor), maps them to hat-coordinates (hat_ctrl), contracts the
tensor to the Lie-coordinate vector h (contract_dynamics!), assembles M_s = Σ_μ h_μ L_μ, and
multiplies in exp(-iM_s). Returns nothing if any generator goes non-finite (so the optimiser can
reject it). nc = dyn.Γ + 1 coefficients per segment, where Γ is the time-truncation order of the
tensor. 
"
function magnus_propagate(dyn, Lbasis, theta, T, nseg, m, omega, Mh, alpha, beta)
    d = size(Lbasis[1], 1); tau = T / nseg; shat = alpha * tau
    nc = dyn.Γ + 1                                   # = maxγ+1, same fix as synthesis
    U = Matrix{ComplexF64}(I, d, d)
    h = zeros(ComplexF64, dyn.dim); Heff = zeros(ComplexF64, d, d)
    for s in 1:nseg
        ts = (s - 1) * tau
        controls = hat_ctrl(fourier_taylor(theta, ts, nc, omega, Mh), alpha, beta)  # nc coeffs
        fill!(h, 0)
        contract_dynamics!(h, dyn.dynamics_tensor, controls, shat)
        all(isfinite, h) || return nothing
        fill!(Heff, 0)
        @inbounds for mu in 1:dyn.dim; Heff .+= h[mu] .* Lbasis[mu]; end
        all(isfinite, Heff) || return nothing
        U = exp(-im .* Heff) * U
    end
    U
end


"
Builds the design sys, unit-normalises the generators forms a cache tag from 
(Nmax, buffer, depth, MDEG, optol), and if the .jld2 is absent it first runs a 
cheap generate_algebra_fast dimension pre-check (skips if dim(g) > PRECOMP_DIM_CAP, 
since the tensor build scales like dim(g)^3), then calls
calculate_tensor_const_degree to construct and save the constant-degree structure tensor 𝒯
It loads the result via load_dynamics_cdeg, collects the orthogonal basis {L_μ},
 and returns (sysd, dyn, Lbasis, dim, α, β, secs)
"
function precompute(Nmax, buffer, depth, optol)
    sysd = build_system(ETA_SR88, Nmax; omega = OMEGA_REL, buffer = buffer)
    # Unit-normalise the generators before building the structure tensor (consistent
    # with the reachability path, App. J): with the raw drift ||A|| ~ omega*Nsim >> 1,
    # the algebra generation is ill-conditioned and the tensor acquires huge/non-finite
    # entries that later overflow in contract_dynamics -> expv. We absorb the scales
    # (alpha,beta) into the segment time and control coefficients at synthesis time.
    alpha = opnorm(Matrix(sysd.A)); beta = opnorm(Matrix(sysd.B))
    An = sparse(sysd.A ./ alpha);  Bn = sparse(sysd.B ./ beta)
    tag   = "Nmax$(Nmax)_bd$(buffer)_d$(depth)_m$(MDEG)_ot$(round(Int, -log10(optol)))_un"
    fname = joinpath(DATA, "magtensor_$(tag).jld2")
    secs  = 0.0
    if !isfile(fname)
        # cheap pre-check of dim(g): skip if the precompute (cost ~ dim(g)^3) is too big
        dimg = generate_algebra_fast([An, Bn], depth; op_rtol = optol, trim_atol = min(optol, 1e-14),
                                     maxelem = PRECOMP_DIM_CAP + 1, verbose = 0).dim
        if dimg > PRECOMP_DIM_CAP
            @printf("    [skip] Nmax=%d buffer=%d depth=%d optol=%.0e: dim(g)=%d > cap %d\n",
                    Nmax, buffer, depth, optol, dimg, PRECOMP_DIM_CAP)
            return nothing
        end
        @printf("    precomputing tensor (dsim=%d, depth=%d, m=%d; dim(g)<=%d; unit-normalised) ...\n",
                sysd.dsim, depth, MDEG, sysd.dsim^2)
        secs = @elapsed calculate_tensor_const_degree(fname, An, Bn, depth, MDEG;
                                                      verbose = 0, optol = optol,
                                                      trim_tol = min(optol, 1e-14))
        @printf("    precompute done in %.1f s -> %s\n", secs, basename(fname))
    end
    obj = load(fname)
    dyn = load_dynamics_cdeg(ComplexF64, obj, depth)
    Lbasis = collect(values(dyn.orthogonal_elements))
    (; sysd, dyn, Lbasis, dim = dyn.dim, alpha, beta, secs)
end


"
The convergence segmentation nseg = max(8, ⌈T/(CONV_SAFETY·π/(‖A‖+2‖B‖))⌉)
"
magnus_nseg(sysd, T) =
    max(8, ceil(Int, T / (CONV_SAFETY * pi / (opnorm(Matrix(sysd.A)) + 2.0 * opnorm(Matrix(sysd.B))))))

"
Same product propagator, but driven by raw per-segment polynomial controls 
(segctrls[s] = [c_0, c_1, …]). It mirrors exactly the generator that the 
Trajectory/expv machinery propagates, so it scores the analytic-gradient solution 
self-consistently in the tensor's own units.
"
function magnus_U(dyn, Lbasis, segctrls, tau, nseg)
    d = size(Lbasis[1], 1); U = Matrix{ComplexF64}(I, d, d)
    h = zeros(ComplexF64, dyn.dim); Heff = zeros(ComplexF64, d, d)
    for s in 1:nseg
        fill!(h, 0); contract_dynamics!(h, dyn.dynamics_tensor, segctrls[s], tau)
        all(isfinite, h) || return nothing
        fill!(Heff, 0)
        @inbounds for mu in 1:dyn.dim; Heff .+= h[mu] .* Lbasis[mu]; end
        all(isfinite, Heff) || return nothing
        U = exp(-im .* Heff) * U
    end
    U
end

"
Collapses each segment's local polynomial to a single PWC midpoint value
d_s(τ/2) = Σ_a c_a (τ/2)^a / a!, so a Magnus pulse can be re-scored on the 
exact static propagator like any other waveform.
"
render_delta(segctrls, tau) =
    [sum(c[a+1] * (tau/2)^a / factorial(a) for a in 0:length(c)-1) for c in segctrls]

"
The unit-normalisation scale maps of ĉ_n = β d_n / α^{n+1} (physical  to
tensor coordinates) and its inverse d_n = ĉ_n α^{n+1} / β. Synthesised coefficients live in
hat-coordinates and must be pushed back through phys_ctrl to mean a physical Δ(t)
"
hat_ctrl(d, a, b)  = [d[n+1] * b / a^(n+1) for n in 0:length(d)-1]
phys_ctrl(c, a, b) = [c[n+1] * a^(n+1) / b for n in 0:length(c)-1]

"
Sets nc = Γ + 1 control coefficients per segment, nseg/τ/ŝ.
Builds the register-basis state-transfer set on the design space: inputs inps = columns of P,
targets tgts = columns of P·G (the gate applied to those inputs). For each pair it forms a
Trajectory from the flattened per-segment parameters [ŝ, c_0, …, c_{Γ}] and accumulates the
cost and analytic gradient:
J = nst - Σ_r Re ⟨φ_r | U | ψ_r⟩ ,   φ_r = (σ_x ⊗ I)|ψ_r⟩,
via cost_trajectory and grad_cost_trajectory. Minimising J drives
Re Tr(G†W) TO d_reg, i.e. the point-target gate. Optimised by L-BFGS with OPT_RESTARTS
restarts; the hat-coordinate solution is mapped back to a physical Δ(t) (phys_ctrl +
render_delta), and the routine reports design_infid = 1 - Favg and the propagation error
prop_err = ‖U_magnus - U_exact‖.
Segment durations are held fixed at τ = T/nseg so the comparison with fixed-T GRAPE is exact, 
even though [dSK] support optimising τ directly. 
seg_generators_finite rejects parameter vectors that produce non-finite generators before propagation; 
a try/catch inside fg! converts LAPACK overflow aborts into the same rejection penalty;
a non-finite trivial seed triggers a fallback to the derivative-free engine.
"
function magnus_synthesize_grad(pc, T)
    sysd, dyn, Lbasis = pc.sysd, pc.dyn, pc.Lbasis
    alpha, beta = pc.alpha, pc.beta
    Gamma = dyn.Γ                 # keep for reference
    nc    = Gamma + 1             # control coefficients per segment = maxγ + 1  (THE FIX)
    nseg  = magnus_nseg(sysd, T); tau = T / nseg
    shat  = alpha * tau
    Hbuf  = zeros(ComplexF64, size(Lbasis[1]))
    # full register-basis state-transfer set on the DESIGN sim space
    Pd, Gd = Matrix(sysd.P), Matrix(sysd.G)
    inps = [ComplexF64.(Pd[:, j])      for j in 1:sysd.dreg]
    tgts = [ComplexF64.(Pd * Gd[:, j]) for j in 1:sysd.dreg]
    nst  = length(inps)
    flatten(x) = begin
        flat = Vector{Float64}(undef, nseg * (nc + 1))
        @inbounds for s in 1:nseg
            flat[(s-1)*(nc+1) + 1] = shat
            flat[(s-1)*(nc+1)+2 : s*(nc+1)] .= @view x[(s-1)*nc+1 : s*nc]
        end
        flat
    end

    seg_generators_finite(x) = begin
        @inbounds for s in 1:nseg
            h = contract_dynamics(dyn.dynamics_tensor, dyn.dim,
                                  collect(view(x, (s-1)*nc+1 : s*nc)), shat)   # nc coeffs
            all(isfinite, h) || return false
            fill!(Hbuf, 0)
            for mu in 1:dyn.dim; Hbuf .+= h[mu] .* Lbasis[mu]; end
            all(isfinite, Hbuf) || return false
        end
        true
    end

    function fg!(F, G, x)
        G === nothing || fill!(G, 0.0)
        seg_generators_finite(x) || return 2.0 * nst          # reject non-finite generators
        flat = flatten(x)
        J = Float64(nst)
        for r in 1:nst
            traj = Trajectory(inps[r], tgts[r], flat, nseg, nc)
            # Belt-and-suspenders: even with finite per-segment generators an
            # intermediate propagated state can overflow; a throw here would abort
            # the whole optimisation inside LAPACK, so trap it and return the same
            # rejection penalty the guard uses.
            try
                if G !== nothing
                    gc = grad_cost_trajectory(traj, dyn)    # per-seg [d/dtau, d/dc0, ...]
                    @inbounds for s in 1:nseg, a in 1:nc
                        G[(s-1)*nc + a] -= real(gc[s][a+1])
                    end
                end
                J -= real(cost_trajectory(traj, dyn))       # reuses cached forward prop
            catch err
                err isa ArgumentError || rethrow(err)        # only swallow chkfinite-type aborts
                G === nothing || fill!(G, 0.0)
                return 2.0 * nst
            end
        end
        isfinite(J) ? J : 2.0 * nst
    end

    x0 = zeros(Float64, nseg * nc)
    for s in 1:nseg
        t0 = (s - 1) * tau
        x0[(s-1)*nc + 1] = beta * (0.1*cos(sysd.omega*t0)) / alpha
        nc >= 2 && (x0[(s-1)*nc + 2] = beta * (-0.1*sysd.omega*sin(sysd.omega*t0)) / alpha^2)
        # x0[(s-1)*nc + 3] stays 0  → degree-2 term starts off
    end

    best = x0; bestJ = fg!(0.0, nothing, x0)
    if !isfinite(bestJ) || bestJ >= 1.5 * nst
        @warn string("Magnus(gradient): the trivial seed already yields a non-finite Magnus ",
                     "generator at Nmax=", sysd.Nmax, ", T=", T, " (likely an ill-conditioned ",
                     "structure tensor from the non-unit-normalised synthesis precompute). ",
                     "Falling back to the derivative-free engine for this point.")
        return magnus_synthesize_nm(pc, T)
    end
    if HAVE_OPTIM
        for r in 0:OPT_RESTARTS
            xs  = r == 0 ? x0 : x0 .+ 0.05 .* randn(MersenneTwister(r), length(x0))
            res = Optim.optimize(Optim.only_fg!(fg!), xs, Optim.LBFGS(),
                                 Optim.Options(iterations = 300, g_tol = 1e-10))
            Optim.minimum(res) < bestJ && (bestJ = Optim.minimum(res); best = Optim.minimizer(res))
        end
    end

    chat_segs = [collect(best[(s-1)*nc+1 : s*nc]) for s in 1:nseg]
    nparams   = nseg * nc
    phys_segs = [phys_ctrl(c, alpha, beta) for c in chat_segs]   # chat -> physical d_n
    delta    = render_delta(phys_segs, tau)                      # physical Delta(t) waveform
    Um       = magnus_U(dyn, Lbasis, chat_segs, shat, nseg)      # normalised tensor, shat time
    Ue       = propagate(sysd, delta, T)                         # exact static, same waveform
    design_infid = Um === nothing ? 1.0 : 1 - metrics(sysd, Um).Favg
    prop_err     = Um === nothing ? Inf : opnorm(Um - Ue)
    (; delta, nseg, nparams, design_infid, prop_err)
end

"
Derivative-free fallback (Fourier control AND Nelder-Mead through magnus_propagate). 
retained for environments where the gradient path misbehaves. Returns the same
record shape so magnus_row is agnostic to which ran.
"
function magnus_synthesize_nm(pc, T)
    sysd, dyn, Lbasis = pc.sysd, pc.dyn, pc.Lbasis
    alpha, beta = pc.alpha, pc.beta
    nseg = magnus_nseg(sysd, T)
    obj(theta) = (U = magnus_propagate(dyn, Lbasis, theta, T, nseg, MDEG, sysd.omega, MH, alpha, beta);
                  U === nothing ? 1.0 : (f = 1 - metrics(sysd, U).Favg; isfinite(f) ? f : 1.0))
    theta0 = zeros(Float64, 1 + 2 * MH); theta0[2] = 0.1
    best, bestf = theta0, obj(theta0)
    for r in 0:OPT_RESTARTS
        th0 = r == 0 ? theta0 : theta0 .+ 0.2 .* randn(MersenneTwister(r), length(theta0))
        res = Optim.optimize(obj, th0, Optim.NelderMead(), Optim.Options(iterations = 400))
        Optim.minimum(res) < bestf && (bestf = Optim.minimum(res); best = Optim.minimizer(res))
    end
    Um = magnus_propagate(dyn, Lbasis, best, T, nseg, MDEG, sysd.omega, MH, alpha, beta)
    Ue = propagate(sysd, sample_delta(best, T, 8nseg, sysd.omega, MH), T)
    (; delta = sample_delta(best, T, nseg, sysd.omega, MH), nseg, nparams = 1 + 2 * MH,
       design_infid = bestf, prop_err = Um === nothing ? Inf : opnorm(Um - Ue))
end

magnus_synthesize(pc, T) = MAGNUS_GRADIENT ? magnus_synthesize_grad(pc, T) : magnus_synthesize_nm(pc, T)

end # module
