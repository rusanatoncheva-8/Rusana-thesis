#  RecoilFreeGate.jl 
"
Summary:

The main purpose of this module is to provide every operaotr, metric, and optimiser used in the rest of the sctipts.
The scripts builds the buffered simulation operators, scores a propagator against the recoil the recoil-free X-gate
by different metrics, some of them used during optimisation, some only as a validation method.
In The ends it syntehises a control pulses by free-segment GRAPE with an exact analytical gradient.

Methods:
Picewise-constant propagation; exact Frechet derivaties of the segment exponential via a Hermitian eigendecomposition
and the Daleckii-Krein divided-difference formula; standard forward/backward GRAPE gradients; multistarts and multigrid optimisation;

Output:
In-memory objects and CSV writers for time-resolved trajectories and method comparisons.
"
module RecoilFreeGate

using LinearAlgebra, Printf, Random, Statistics

const HAVE_OPTIM = try
    @eval import Optim
    true
catch
    false
end

# Sr-88 operating point
const ETA_SR88    = 0.9648
const OMEGA_REL   = 1.6667
const DEMO_BUFFER = 12

export ETA_SR88, OMEGA_REL, DEMO_BUFFER, HAVE_OPTIM
export build_system, propagate, seg_count, fock_blocks, thermal_pops, recommend_nmax,
       phase_spread, metrics, atomic_channel_fidelity, manifold_factorization
export cost_grad!, transfer_targets, cost_grad_transfer!, upsample, opt_gate, opt_transfer, synthesize
export grade_nonzero_fraction
export compare_methods, write_comparison_csv
export thermal_gate_fidelity, thermal_heating
export floor_scan, minimum_time_scaling, _linfit, _crossing
export propagate_record, pulse_phase, state_evolution, recoil_trace, write_csv, export_pulse_csv

#  Physical model and operators
"Annihilation operator on N+1 Fock levels:  a|n> = sqrt(n)|n-1>."
function ladder(N)
    a = zeros(ComplexF64, N+1, N+1)
    for n in 1:N
        a[n, n+1] = sqrt(n)
    end
    a
end

"""
Sets up the static-frame Hamiltonian components A (drift) and B (control),
 where H(t) = A + Δ(t)B. Builds the internal-external coupling operator K, 
 the target gate G, and the projection matrix P used to extract the computational register.
 Note that the Nsim is used construct the ladder and number operators.
"""
function build_system(eta, Nmax; omega = OMEGA_REL, buffer = DEMO_BUFFER)
    Nsim = Nmax + buffer
    dmot = Nsim + 1
    a    = ladder(Nsim)
    x    = a + a'
    D    = exp(im * eta * x)                  # buffered displacement operator
    num  = diagm(0 => ComplexF64.(0:Nsim))
    I2   = Matrix{ComplexF64}(I, 2, 2)
    Im   = Matrix{ComplexF64}(I, dmot, dmot)
    sx   = ComplexF64[0 1; 1 0]
    sy   = ComplexF64[0 -im; im 0]
    sz   = ComplexF64[1 0; 0 -1]
    p10  = ComplexF64[0 0; 1 0]               # |1><0|
    p01  = ComplexF64[0 0; 0 0]; p01[1,2] = 1 # |0><1|
    K    = kron(p10, D) + kron(p01, D')       # internal-external coupling
    N    = kron(I2, num)
    A    = omega * N + 0.5 * K                # fixed drift
    B    = kron(sz, Im)                       # control operator sigma_z
    dsim = 2 * dmot
    dreg = 2 * (Nmax + 1)
    # Register embedding P (dsim x dreg), atom outer factor
    P = zeros(ComplexF64, dsim, dreg)         # Dsim rows and dreg columns
    # Double loop iterating over the atom and motional state, linking a register state to its corresponding simulation state.
    for atom in 0:1, n in 0:Nmax
        P[atom*dmot + n + 1, atom*(Nmax+1) + n + 1] = 1
    end
    G  = kron(sx, Matrix{ComplexF64}(I, Nmax+1, Nmax+1))    # point target sigma_x (x) I_reg
    Q  = P * G' * P'                          # Projection
    # A list of targets gets, one for each individual motional level; put 1 only ta the diagonal positions;
    Gn = [kron(sx, (e = zeros(ComplexF64, Nmax+1, Nmax+1); e[n+1,n+1] = 1; e)) for n in 0:Nmax]
    # Similarly to Gn, but a list for the embeded co-state matrices for the individual Gn target (used for the analytical gradient in GRAPE)
    Qn = [P * Gn[n+1]' * P' for n in 0:Nmax]
    Nreg = kron(I2, diagm(0 => ComplexF64.(0:Nmax)))  # Motional number on the restricted space
    wNsim = omega * N
    (; A, B, K, P, G, Q, Gn, Qn, Nreg, wNsim,
       dsim, dmot, dreg, Nmax, Nsim, eta, omega,
       sx, sy, sz, I2, Im)
end

"Computes the total time-evolution operator (propagator) U.
Uses a piecewise-constant approximation, applying the static-frame
Hamiltonian segment by segment: U = ∏_j exp(-i(A+δ_jB) dt)."
function propagate(sys, delta, T)
    S = length(delta); dt = T/S
    U = Matrix{ComplexF64}(I, sys.dsim, sys.dsim)
    for j in 1:S
        U = exp(-im * (sys.A + delta[j]*sys.B) * dt) * U
    end
    U
end

"A heuristic helper function to determine the number of segments S needed for a given gate time T"
seg_count(T) = max(8, round(Int, 2.2 * T))

# Figures of merrit.
"Extracts the diagonal 2 × 2 atomic blocks from the register block W for each Fock state n ∈ [0, N_max]"
function fock_blocks(sys, W)
    nb = sys.Nmax + 1
    [W[[a*nb + n + 1 for a in 0:1], [a*nb + n + 1 for a in 0:1]] for n in 0:sys.Nmax]
end

"Generates a normalised thermal probability distribution for a harmonic oscillator with average phonon number ̄ n"
function thermal_pops(nbar, levels)
    if nbar <= 0
        p = zeros(Float64, levels); p[1] = 1.0; return p
    end
    r = nbar/(nbar+1)
    p = [r^n for n in 0:levels-1]
    p ./ sum(p)
end

"Calculates the minimum register size N_max required to capture a specific percentage  of the thermal population"
function recommend_nmax(nbar; coverage = 0.999)
    nbar <= 0 && return 0
    r = nbar/(nbar+1)
    max(0, ceil(Int, log(1 - coverage)/log(r)) - 1)
end

"Calculates the circular standard deviation of the phases across different Fock blocks, ignoring blocks with amplitudes below ampthr"
function phase_spread(cvec; ampthr = 1e-3)
    # Filter out any complex number c which absolute values is below a threshold.
    keep = [c for c in cvec if abs(c) > ampthr]
    isempty(keep) && return (; circstd = NaN, phases = Float64[])
    phases = angle.(keep)                               # Extracts the amplitudes
    # Normal average or std cannot be used on angles. Instead every angle is treated as a unit on the unit circle.
    Rbar = abs(sum(cis, phases)) / length(phases)       # If all phases add up perfectly, then Rbar=1.0.
    (; circstd = sqrt(max(0.0, -2*log(max(Rbar, eps())))), phases)
end

"""
Returns a A NamedTuple containing process fidelity F_{pro}, leakage-aware average
gate fidelity F_{avg}, manifold fidelities, structural recoil metrics R_{grade}, commHS), 
and operational heating Δ ̄ n, Pret).

c_n coefficients define the alignment of the atomic block sector n with sigma_x flip.
g is the sum of those coefficients.

Fpro = |g|²/d² is the process fidelity to the point target which is global-phase invariant via
|·|²; It penalises phase disagreement between Fock sectors, because the c_n  coefficients add as
complex vectors in g.

Favg = (|g|² + Tr(W†W))/(d(d+1)) is the leakage-aware (Pedersen) average gate
fidelity; Tr(W†W) < d whenever population leaks, so this one number charges for flip
quality, phase uniformity, and leakage. 

Fman_aligned = (Σ|c_n|)²/d²  sums the magnitudes of the c_n coefficients, instead of the complex numbers directly,
making it ignore pahse differences per Fock sector. It is used in the diagnostics only.

Fman_block = Σ|c_n|² / (4(Nmax+1)) is the smoot maniforld objective taht sums the squares of |c_n|; one of the biggest
advantages is the clean deriavtive which can be used in the GRAPE optimisation; it ignores differences in the phase
per Fock state; If this gate reaches 1, the gate would be recoil-free and leak-free.

Rgrade = (‖W‖²_HS - Σ‖W_{nn}‖²_HS)/‖W‖²_HS is the fraction of register weight in off-diagonal (phonon-changing) 
blocks; the metric is dimensionless, and therefore comparable across Nmax. The Hilber-Schmitd norma is used.

commHS = ‖[W,N_reg]‖_HS = √(Σ (m-n)² ‖W_{mn}‖²_HS) is the smooth, differentiable recoil measure used as the
optimiser penalty; weights large phonon jumps quadratically.

rM is the grade-≠0 fraction of M = i·log(unitary part of W); the literal generator statement, computed via 
SVD-polar and eigen-log. This serves only as endpoint diagnostic due to possible problems with the logarithm .

dnbar, Pret are operational heating Δ⟨N⟩ and motional-return probability for a thermal n̄ input with a maximally
mixed atom (only when nbar>0).

leak = 1 - Tr(W†W)/d
"""
function metrics(sys, U; nbar = 0.0)
    W = sys.P' * U * sys.P
    d = sys.dreg
    g = tr(sys.G' * W)
    blks = fock_blocks(sys, W)
    cvec = [tr(sys.sx * b) for b in blks]
    Fpro = abs2(g) / d^2
    survive = real(tr(W' * W))
    Favg = (abs2(g) + survive) / (d*(d+1))
    Fman_aligned = sum(abs, cvec)^2 / d^2
    Fman_block   = sum(abs2, cvec) / (4*(sys.Nmax+1))
    # Recoil R1
    totHS  = real(tr(W'*W))
    diagHS = sum(real(tr(b'*b)) for b in blks)
    Rgrade = totHS < eps() ? 0.0 : (totHS - diagHS)/totHS
    # Recoil R2
    C = W*sys.Nreg - sys.Nreg*W
    commHS = sqrt(real(tr(C'*C)))
    # Recoil R3 
    rM = try
        F = svd(W); Wu = F.U * F.Vt
        e = eigen(Wu); M = e.vectors * Diagonal(angle.(e.values)) * e.vectors'
        M = (M + M')/2
        grade_nonzero_fraction(M, sys.Nmax+1)   # W (hence M) is register-sized; grade directly
    catch
        NaN
    end
    # Operational heating R4
    dnbar = NaN; Pret = NaN
    if nbar > 0
        nb = sys.dmot
        preg = zeros(Float64, nb); preg[1:sys.Nmax+1] .= thermal_pops(nbar, sys.Nmax+1)
        Nop  = sys.wNsim / sys.omega
        rho  = kron(ComplexF64[0.5 0; 0 0.5], ComplexF64.(diagm(0 => preg)))
        rho_o = U * rho * U'
        dnbar = real(tr(Nop*rho_o) - tr(Nop*rho))
        Pr = 0.0
        for n in 0:sys.Nmax
            preg[n+1] == 0 && continue
            e = zeros(ComplexF64, nb, nb); e[n+1,n+1] = 1
            out = U * kron(ComplexF64[0.5 0; 0 0.5], e) * U'
            Pr += preg[n+1] * real(out[n+1,n+1] + out[nb+n+1, nb+n+1])
        end
        Pret = Pr
    end
    leak = 1 - survive/d
    ps = phase_spread(cvec)
    (; Fpro, Favg, Fman_aligned, Fman_block,
       Rgrade, commHS, rM, dnbar, Pret, leak,
       phi_circstd = ps.circstd, cabs = abs.(cvec), g)
end

"Calculates the fraction of an operator that lies outside the grade-zero (recoil-free) subspace.
It calculates the suqared absolute value of Mij of the complex matrix element that represent the transition
probability from state i to state j."
function grade_nonzero_fraction(M, nb)
    n = size(M, 1); tot = 0.0; g0 = 0.0
    @inbounds for i in 1:n, j in 1:n
        v = abs2(M[i,j]); tot += v
        ((i-1) % nb == (j-1) % nb) && (g0 += v)
    end
    tot < eps() ? 0.0 : (tot - g0)/tot
end

"""
Calculates the leakage-aware average gate fidelity of the atomic channel to σ_x,
assuming the motional state is prepared in the thermal distribution rho_m.
I traces out the motion to get the atomic channel with 2×2 Kraus operators 
A_{mn} = √p_n ⟨m|U|n⟩, then the Pedersen average gate fidelityto σ_x: 
(Σ|Tr(σ_x† A_{mn})|² + Σ Tr(A_{mn}†A_{mn}))/6
"""
function atomic_channel_fidelity(sys, U, rho_m)
    nb = sys.dmot; da = 2
    s1 = 0.0; s2 = 0.0
    for n in 0:sys.Nmax
        p = rho_m[n+1]; p == 0 && continue
        for m in 0:sys.Nsim
            A = sqrt(p) * U[[m+1, nb+m+1], [n+1, nb+n+1]]   # 2x2 atomic Kraus block
            s1 += abs2(tr(sys.sx' * A))
            s2 += real(tr(A' * A))
        end
    end
    (s1 + s2) / (da*(da+1))
end

"A diagnostic tool that checks how well the register block factorizes into an atomic flip 
and a motional phase shift. Returns the per-Fock phases and the reconstruction error."
function manifold_factorization(sys, U)
    W = sys.P' * U * sys.P
    blks = fock_blocks(sys, W)
    cvec = [tr(sys.sx * b) for b in blks]
    phis = angle.(cvec)
    recon = sum(abs2, [blks[n+1] - cis(phis[n+1])*sys.sx for n in 0:sys.Nmax]) /
            max(eps(), sum(abs2, blks))
    (; phis, recon_err = recon)
end

#  GRAPE engine 
"""
Computes per-segment propagators U_j, their exact analytic Fréchet derivatives with 
respect to the control amplitude ∂U_j/∂ δ_j), and the forward/backward propagation products.
Instead of general matrix exponentials, it leverages the fact that H_j = A + δ_jB 
is Hermitian. It uses Hermitian eigen-decomposition (H_j = V Λ V^†) to compute the 
derivatives via the spectral divided-difference formula. This is done to  minimise memory allocation
compared to standard augmented-exponential methods.
"""
function _segments(sys, T, delta)
    S = length(delta); dt = T/S; n = sys.dsim
    Us  = Vector{Matrix{ComplexF64}}(undef, S)
    dUs = Vector{Matrix{ComplexF64}}(undef, S)
    Bdt = -im .* Matrix(sys.B) .* dt          #Ddirection E for the Fréchet derivative
    for j in 1:S
        Hj = Hermitian(sys.A .+ delta[j] .* sys.B)
        F  = eigen(Hj)                         # Real eigenvalues λ, unitary V
        λ  = F.values; V = F.vectors
        ph = exp.((-im*dt) .* λ)               # e^{-iλ dt}
        Us[j] = V * Diagonal(ph) * V'
        # Divided-difference matrix Λ̄_{ab} for f(z)=e^z at z_a=-iλ_a dt
        z = (-im*dt) .* λ
        Λ̄ = Matrix{ComplexF64}(undef, n, n)
        @inbounds for b in 1:n, a in 1:n
            if abs(λ[a]-λ[b]) < 1e-12
                Λ̄[a,b] = ph[a]                 # derivative of e^z at coincident eigenvalues
            else
                Λ̄[a,b] = (ph[a]-ph[b])/(z[a]-z[b])
            end
        end
        Etil = V' * Bdt * V                     # rotate the direction into the eigenbasis
        dUs[j] = V * (Λ̄ .* Etil) * V'
    end
    Fp = Vector{Matrix{ComplexF64}}(undef, S+1)
    Bk = Vector{Matrix{ComplexF64}}(undef, S+1)
    Fp[1] = Matrix{ComplexF64}(I, n, n)
    for j in 1:S; Fp[j+1] = Us[j] * Fp[j]; end
    Bk[S+1] = Matrix{ComplexF64}(I, n, n)
    for j in S:-1:1; Bk[j] = Bk[j+1] * Us[j]; end
    (; Us, dUs, Fp, Bk, dt, S)
end

"""
Evaluates the primary objective function (either the point target or the manifold target) 
and simultaneously computes its exact analytic gradient with respect to the control parameters.
Calculates the infidelity 1-F and applies an exact penalty for structural recoil
(R = ||[W, N_{reg}]||^2 / d_{reg}) using the weights lambda and mu. 
If Gout is provided, it populates it with the exact gradients using the Fréchet derivatives from _segments.
"""
function cost_grad!(Gout, sys, x; objective = :point, lambda = 0.0, mu = 0.0)
    T = x[1]; delta = @view x[2:end]
    seg = _segments(sys, T, delta); U = seg.Fp[end]
    W = sys.P' * U * sys.P; d = sys.dreg    # Shrinks down the simulation unitary to the computational register.
    # gate value + co-matrix
    if objective === :point
        g = tr(sys.G' * W); F = abs2(g)/d^2
        Qg = conj(g) * sys.Q    # The analytical derivative of the smooth metric.
        gatefac = 2/d^2
    else
        cvec = [tr(sys.Gn[n+1]' * W) for n in 0:sys.Nmax]
        F = sum(abs2, cvec)/(4*(sys.Nmax+1))
        Qg = sum(conj(cvec[n+1]) * sys.Qn[n+1] for n in 0:sys.Nmax)     # Again an anlytical derivative.
        gatefac = 2/(4*(sys.Nmax+1))
    end
    # Recoil term
    C  = W*sys.Nreg - sys.Nreg*W
    R  = real(tr(C'*C))/d   # Squared Frobenius norm of the commutator.
    QR = sys.P * (sys.Nreg*C' - C'*sys.Nreg) * sys.P'
    Jc = (1 - F) + lambda*R + 0.5*mu*R^2        # The linear term dominates when R is small
    if Gout !== nothing
        S = seg.S; wR = (lambda + mu*R)
        Qco = (-gatefac) .* Qg .+ (2*wR/d) .* QR
        gd = zeros(Float64, S); gT = 0.0
        for j in 1:S
            M = seg.Fp[j] * Qco * seg.Bk[j+1]      # co-matrix (computed once per segment)
            gd[j] = real(tr(M * seg.dUs[j]))
            Hj = sys.A + delta[j]*sys.B
            gT += real(tr(M * ((-im/S) * Hj * seg.Us[j])))
        end
        Gout[1] = gT
        Gout[2:end] .= gd
    end
    Jc
end

"Formulates a relaxed transfer objective where the atomic state must be flipped, but the final motional 
state is not restricted within the computational register ."
function transfer_targets(sys)
    nb = sys.dmot
    ket(a) = (v = zeros(ComplexF64, 2); v[a+1] = 1; v)
    inputs = [ket(0), ket(1), (ket(0)+ket(1))/sqrt(2), (ket(0)+im*ket(1))/sqrt(2)]
    motion0 = (v = zeros(ComplexF64, nb); v[1] = 1; v)
    Ireg = zeros(ComplexF64, nb, nb); for n in 0:sys.Nmax; Ireg[n+1,n+1] = 1; end
    out = NamedTuple[]
    for a in inputs
        psi = kron(a, motion0)
        chi = sys.sx * a
        Theta = kron(chi*chi', Ireg)
        push!(out, (; psi, Theta))
    end
    out
end

"Value/gradient for the recoil-agnostic transfer objective.
P = (1/N_t) Σ_k ⟨ψ_k|U†Θ_k U|ψ_k⟩, cost J = (1-P) + λR + ½μR². The transfer
gradient is assembled from rank-1 pieces acc = Σ_k (Θ_k U ψ_k)(ψ_k†) (a d×d outer
product per target), giving the transfer co-matrix M_trans = -(2/N_t)(Bk[j+1]† acc)†, plus
the same recoil co-matrix M_rec = Fp[j](2w_R/d Q_R)Bk[j+1]
"
function cost_grad_transfer!(Gout, sys, x, targets; lambda = 0.0, mu = 0.0)
    T = x[1]; delta = @view x[2:end]
    seg = _segments(sys, T, delta); U = seg.Fp[end]
    W = sys.P' * U * sys.P; d = sys.dreg
    P = 0.0
    for t in targets
        v = U * t.psi
        P += real(v' * t.Theta * v)
    end
    P /= length(targets)
    C = W*sys.Nreg - sys.Nreg*W
    R = real(tr(C'*C))/d
    QR = sys.P * (sys.Nreg*C' - C'*sys.Nreg) * sys.P'
    Jc = (1 - P) + lambda*R + 0.5*mu*R^2
    if Gout !== nothing
        S = seg.S; wR = (lambda + mu*R)
        gd = zeros(Float64, S)
        for j in 1:S
            acc = zeros(ComplexF64, sys.dsim, sys.dsim)
            for t in targets
                acc .+= (t.Theta * (seg.Fp[end] * t.psi)) * (t.psi')   # rank-1 per target
            end
            Mtrans = -(2/length(targets)) .* (seg.Bk[j+1]' * acc)'      # transfer part
            Mrec   = seg.Fp[j] * ((2*wR/d) .* QR) * seg.Bk[j+1]
            gd[j] = real(tr((Mtrans) * seg.dUs[j])) + real(tr(Mrec * seg.dUs[j]))
        end
        Gout[1] = 0.0
        Gout[2:end] .= gd
    end
    Jc
end

" The optimizer first looks for a rough pulse using only a few thick segments.
The upsample function takes that small number of segments pulse and  maps it onto a a bigger segment grid.
The optimizer is then run again on the new grid."
function upsample(delta, Snew)
    S = length(delta); S == Snew && return copy(delta)
    told = [(j-0.5)/S for j in 1:S]
    tnew = [(j-0.5)/Snew for j in 1:Snew]
    out = similar(delta, Snew)
    for (i, t) in enumerate(tnew)
        if t <= told[1]; out[i] = delta[1]
        elseif t >= told[end]; out[i] = delta[end]
        else
            k = searchsortedlast(told, t)
            f = (t - told[k])/(told[k+1] - told[k])
            out[i] = (1-f)*delta[k] + f*delta[k+1]
        end
    end
    out
end

"Optimiser for a fixed total gate time. If the optim package is available, uses L-BFGS."
function _optimise(costfun!, delta0, T, iters)
    if HAVE_OPTIM
        x = vcat(T, delta0)
        function fg!(F, G, xv)
            Gf = G === nothing ? nothing : zeros(Float64, length(xv))
            J = costfun!(Gf, vcat(xv[1], @view xv[2:end]))
            if G !== nothing
                G[1] = 0.0                      # hold T fixed
                G[2:end] .= @view Gf[2:end]
            end
            J
        end
        res = Optim.optimize(Optim.only_fg!(fg!), x, Optim.LBFGS(),
                             Optim.Options(iterations = iters, g_tol = 1e-12, f_reltol = 1e-14))
        return Optim.minimizer(res)[2:end]
    else
        d = copy(delta0); m = zero(d); v = zero(d); b1 = 0.9; b2 = 0.999; a = 0.05
        for it in 1:iters
            Gf = zeros(Float64, length(d)+1)
            costfun!(Gf, vcat(T, d))
            gd = @view Gf[2:end]
            @. m = b1*m + (1-b1)*gd
            @. v = b2*v + (1-b2)*gd^2
            mh = m ./ (1-b1^it); vh = v ./ (1-b2^it)
            @. d -= a * mh / (sqrt(vh) + 1e-12)
        end
        return d
    end
end

# Full unitary optimisation
opt_gate(sys, delta0, T, iters; objective = :point, lambda = 0.0, mu = 0.0) =
    _optimise((G, x) -> cost_grad!(G, sys, x; objective, lambda, mu), delta0, T, iters)

opt_transfer(sys, delta0, T, iters, targets; lambda = 0.0, mu = 0.0) =
    _optimise((G, x) -> cost_grad_transfer!(G, sys, x, targets; lambda, mu), delta0, T, iters)

"""
Uses a multigrid approach: it generates several random coarse-grained pulses, optimizes them to find the
best basin of attraction, and then upsamples the winner to the full resolution S for final refinement.
"""
function synthesize(sys, T, S; restarts = 4, itc = 300, itf = 700, seed = 1,
                    objective = :point, lambda = 0.0, mu = 0.0)
    Sc = max(50, S ÷ 4)
    score(d) = (m = metrics(sys, propagate(sys, d, T)); objective === :point ? m.Fpro : m.Fman_block)
    best = nothing; bestF = -Inf
    for r in 1:restarts
        d0 = 0.7 .* randn(MersenneTwister(seed + r), Sc)
        d  = opt_gate(sys, d0, T, itc; objective, lambda, mu)
        F = score(d); F > bestF && (bestF = F; best = d)
    end
    for Snew in (max(Sc, S ÷ 2), S)
        best = opt_gate(sys, upsample(best, Snew), T, itf; objective, lambda, mu)
    end
    best
end

"
Packages a pulse's metric into a comparison record.
"
function _row(method, sys, U, T, nseg, nparams, secs, delta)
    m = metrics(sys, U; nbar = 0.0)
    (; method, nparams, nseg,
       infid_avg = 1 - m.Favg, infid_pro = 1 - m.Fpro,
       Rgrade = m.Rgrade, leak = m.leak,
       peakD = delta === nothing ? NaN : maximum(abs, delta), secs)
end

"""
Comparison at fixed (Nmax, T, buffer):
  GRAPE_free    - free-segment GRAPE, exact static prop, analytic gradient
"""
function compare_methods(sys, T; grape_restarts = 4, objective = :point, lambda = 0.0,
                         verbose = true)
    rows = NamedTuple[]
    log(r) = verbose && @printf("    [%-15s] 1-Favg=%.3e  Rgrade=%.2e  leak=%.2e  (%.1fs)\n",
                                r.method, r.infid_avg, r.Rgrade, r.leak, r.secs)
    verbose && println("    running GRAPE_free ...")
    t = @elapsed (dG = synthesize(sys, T, seg_count(T); restarts = grape_restarts, itc = 300, itf = 700, objective, lambda))
    r = _row("GRAPE_free", sys, propagate(sys, dG, T), T, seg_count(T), seg_count(T), t, dG); push!(rows, r); log(r)
    rows
end

function write_comparison_csv(path, rows)
    open(path, "w") do io
        println(io, "method,nparams,nseg,infid_avg,infid_pro,Rgrade,leak,peakD,secs")
        for r in rows
            println(io, @sprintf("%s,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6g,%.3f",
                    r.method, r.nparams, r.nseg, r.infid_avg, r.infid_pro, r.Rgrade, r.leak, r.peakD, r.secs))
        end
    end
    path
end

#  Thermal figure of merit
"""
Computes the thermally-weighted average gate fidelity by evaluating the operation 
across the initial thermal distribution P_n(̄ n) of the harmonic oscillator.
"""
function thermal_gate_fidelity(sys, U; nbar = 0.5)
    W = sys.P' * U * sys.P; blks = fock_blocks(sys, W)
    p = thermal_pops(nbar, sys.Nmax+1); Fth = 0.0
    for n in 0:sys.Nmax
        Wn = blks[n+1]; cn = tr(sys.sx * Wn)
        Fth += p[n+1] * (abs2(cn) + real(tr(Wn'*Wn)))/6
    end
    Fth
end

"Calculates the operational heating (the change in average phonon number Δ ⟨N⟩ and tracks 
how much population leaks entirely outside the simulated computational register."
function thermal_heating(sys, U; nbar = 0.5)
    nb = sys.dmot; Nop = sys.wNsim/sys.omega
    rho_a = ComplexF64[0.5 0; 0 0.5]
    preg = thermal_pops(nbar, sys.Nmax+1)
    plev = zeros(Float64, nb); plev[1:sys.Nmax+1] .= preg
    rho = kron(rho_a, ComplexF64.(diagm(0 => plev))); rho_o = U*rho*U'
    dnbar = real(tr(Nop*rho_o) - tr(Nop*rho))
    Pret = 0.0
    for n in 0:sys.Nmax
        preg[n+1] == 0 && continue
        e = zeros(ComplexF64, nb, nb); e[n+1,n+1] = 1
        out = U * kron(rho_a, e) * U'
        Pret += preg[n+1]*real(out[n+1,n+1] + out[nb+n+1,nb+n+1])
    end
    (; dnbar, Pret, outside_reg_pop = (nbar/(nbar+1))^(sys.Nmax+1))
end


#  Floor / minimum-time scaling
_linfit(x, y) = (n = length(x); sx = sum(x); sy = sum(y); sxx = sum(abs2, x); sxy = sum(x .* y);
                 b = (n*sxy - sx*sy)/(n*sxx - sx^2); (a = (sy - b*sx)/n, b = b))

function _crossing(Ts, infids, target)
    for i in eachindex(Ts)
        if infids[i] <= target
            i == 1 && return Ts[1]
            y1 = log(infids[i-1]); y2 = log(infids[i])
            abs(y2-y1) < eps() && return Ts[i]
            return Ts[i-1] + (log(target)-y1)/(y2-y1)*(Ts[i]-Ts[i-1])
        end
    end
    NaN
end

"""
 Sweeps the gate duration T. It starts with a long time guess, progressively shrinks 
 it, and uses the previous pulse as a "warm start" for the next optimization until 
 the fidelity crosses a failure threshold or plateaus.
"""
function floor_scan(sys; T0 = 30.0 + 25.0*sys.Nmax, growth = 1.22, maxsteps = 8,
                    band = 3.0, window = 3, success = 1e-6, restarts = 2, itc = 200, itf = 400,
                    nbar = 0.0, verbose = true)
    Ts = Float64[]; infids = Float64[]; leaks = Float64[]
    prevd = nothing; floorI = Inf; floorT = T0
    for s in 1:maxsteps
        T = round(T0*growth^(s-1), digits = 1); S = seg_count(T)
        tic = time()
        cands = Vector{Vector{Float64}}()
        prevd !== nothing && push!(cands, opt_gate(sys, upsample(prevd, S), T, itf; objective = :point))  # warm start
        push!(cands, synthesize(sys, T, S; restarts = (s == 1 ? restarts : max(1, restarts-1)), itc, itf, objective = :point))
        Fs = [metrics(sys, propagate(sys, c, T)).Favg for c in cands]
        d = cands[argmax(Fs)]
        t = time() - tic
        m = metrics(sys, propagate(sys, d, T); nbar); inf = 1 - m.Favg
        push!(Ts, T); push!(infids, inf); push!(leaks, m.leak); prevd = d
        verbose && @printf("    OmT=%-6.1f S=%-4d 1-Favg=%-10.2e leak=%-10.2e (%.1fs)\n", T, S, inf, m.leak, t)
        inf < floorI && (floorI = inf; floorT = T)
        inf < success && break
        # windowed floor cap: stop once the last `window` points sit within `band` of their min
        # (robust to the floor's natural shot-to-shot wobble, unlike a 2-point ratio test)
        if length(infids) >= window
            w = infids[end-window+1:end]
            maximum(w)/minimum(w) < band && break
        end
    end
    (; Ts, infids, leaks, floorI, floorT)
end

"Executes floor_scan systematically over various motional limits N_max to extract the scaling laws governing the required pulse duration."
function minimum_time_scaling(; Nmaxs = 1:3, buffer = DEMO_BUFFER, eta = ETA_SR88, omega = OMEGA_REL, kwargs...)
    rows = NamedTuple[]
    for Nmax in Nmaxs
        sys = build_system(eta, Nmax; omega, buffer)
        sc = floor_scan(sys; kwargs...)
        rows = push!(rows, (; Nmax,
            t3 = _crossing(sc.Ts, sc.infids, 1e-3), t4 = _crossing(sc.Ts, sc.infids, 1e-4),
            floorI = sc.floorI, floorT = sc.floorT, Ts = sc.Ts, infids = sc.infids, leaks = sc.leaks))
    end
    rows
end

#  Time-resolved diagnostics and  CSV export
"Returns the system's state propagator at every individual time step, not just the final endpoint."
function propagate_record(sys, delta, T)
    S = length(delta); dt = T/S; U = Matrix{ComplexF64}(I, sys.dsim, sys.dsim)
    Ulist = Vector{Matrix{ComplexF64}}(undef, S+1); Ulist[1] = copy(U)
    for j in 1:S
        U = exp(-im*(sys.A + delta[j]*sys.B)*dt) * U; Ulist[j+1] = copy(U)
    end
    (; ts = collect(0:S) .* dt, Ulist)
end

"Integrates the control detuning Δ(t) to yield the cumulative accumulated phase ϕ(t)."
function pulse_phase(delta, T)
    S = length(delta); dt = T/S; phase = zeros(Float64, S+1)
    for j in 1:S; phase[j+1] = phase[j] + delta[j]*dt; end
    phase
end

"Atom-|1> population, <N>(t), and register Fock populations P_n(t) along a trajectory."
function state_evolution(sys, Ulist, psi0)
    nb = sys.dmot; P1 = Float64[]; Navg = Float64[]; Pn = [Float64[] for _ in 0:sys.Nmax]
    for U in Ulist
        psi = U * psi0
        push!(P1,   sum(abs2(psi[nb + m + 1]) for m in 0:nb-1))
        push!(Navg, sum(m*(abs2(psi[m+1]) + abs2(psi[nb+m+1])) for m in 0:nb-1))
        for n in 0:sys.Nmax; push!(Pn[n+1], abs2(psi[n+1]) + abs2(psi[nb+n+1])); end
    end
    (; P1, Navg, Pn)
end

"Running recoil  fraction Rgrade(t) of the register block of U(t)."
function recoil_trace(sys, Ulist)
    out = Float64[]
    for U in Ulist
        W = sys.P' * U * sys.P; blks = fock_blocks(sys, W)
        tot = real(tr(W'*W)); diagw = sum(real(tr(b'*b)) for b in blks)
        push!(out, tot < eps() ? 0.0 : (tot - diagw)/tot)
    end
    out
end

function write_csv(path, header, cols)
    open(path, "w") do io
        println(io, join(header, ","))
        N = length(cols[1])
        for i in 1:N
            println(io, join([@sprintf("%.10g", cols[c][i]) for c in eachindex(cols)], ","))
        end
    end
    path
end

"""
 Compiles all time-resolved dynamics for a given control pulse into a  CSV for external plotting and analysis.
 """
function export_pulse_csv(sys, delta, T, path)
    rec = propagate_record(sys, delta, T); ph = pulse_phase(delta, T); nb = sys.dmot
    ket(a, n) = (v = zeros(ComplexF64, sys.dsim); v[a*nb + n + 1] = 1; v)
    evg = state_evolution(sys, rec.Ulist, ket(0,0))
    eve = state_evolution(sys, rec.Ulist, ket(0,1))
    rg  = recoil_trace(sys, rec.Ulist)
    S = length(delta); dstep = [k <= S ? delta[k] : delta[S] for k in 1:S+1]
    header = vcat(["t","delta","phase","P1_g","N_g","P1_e1","N_e1","Rgrade"], ["Pn$n" for n in 0:sys.Nmax])
    cols = Any[rec.ts, dstep, ph, evg.P1, evg.Navg, eve.P1, eve.Navg, rg]
    for n in 0:sys.Nmax; push!(cols, eve.Pn[n+1]); end
    write_csv(path, header, cols)
end

end # module