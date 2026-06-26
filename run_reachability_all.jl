#  run_reachability_all.jl   
"
Summary:

The main purpose of this module is check if a target gate lies in the Lie Algebra genrated by the system'
operators, and if the algebra fill the available operaotr space.

Two different studies are performed:
Study 1 (tri_*) with three generators {N, K, σ_z}: the operator-content upper
bound on reachability.
Study 2 (ab_*) with two generators {A, B} with A = ωN + ½K, B = σ_z: the algebra
the single detuning control H(t)=A+Δ(t)B.



Methods: 
Closed-form displacement matrix elements; sparse Lie closure via generate_algebra_fast;
least-squares target residual vs commutator depth; an operating-tolerance rule and a plateau;
extended precision (Double64) for large registers.

Outputs:
reach_optol_sweep/curves/operating.csv,
reach_saturation.csv, reach_depth_curve.csv,
reach_g0_structure.csv, reach_carrier*.csv. 
Study-2: 
the reach_AB_* analogues plus a reach_AB_certificate.csv.
"



include(joinpath(@__DIR__, "..", "src", "RecoilFreeGate.jl"))
using .RecoilFreeGate                      # ETA_SR88, OMEGA_REL
using MagnusTensor.LieAlgebraUtils         # generate_algebra_fast
using LinearAlgebra, SparseArrays, Printf

const DATA = joinpath(@__DIR__, "..", "data"); mkpath(DATA)


const RUN_TRI    = true      # Study 1: {N, K, sigma_z}
const RUN_AB     = true      # Study 2: {A, B}
const QUICK      = false

# Nmax set (shared by both studies). ARGS override; else default below.
function parse_nmax_args(args)
    isempty(args) && return Int[]
    vals = Int[]
    for a in args
        s = strip(a); isempty(s) && continue
        if occursin(":", s)
            r = parse.(Int, split(s, ":")); append!(vals, collect(r[1]:r[end]))
        else
            for tok in split(s, ","); t = strip(tok); isempty(t) && continue; push!(vals, parse(Int, t)); end
        end
    end
    sort!(unique(vals))
end
const _ARGN      = parse_nmax_args(ARGS)
const NMAXS      = isempty(_ARGN) ? (QUICK ? [2, 3, 4] : [2, 3, 4, 5, 6, 7]) : _ARGN
const NMAX_DEPTH = (5 in NMAXS) ? 5 : maximum(NMAXS)   # register size for the depth-vs-eta studies

# Double64 for large Nmax (conditioning of the closure)
HAVE_D64 = false
try
    @eval using DoubleFloats
    global HAVE_D64 = true
catch
    @warn "DoubleFloats not installed -- Nmax>=7 stays Float64 (may not close cleanly). Add it: import Pkg; Pkg.add(\"DoubleFloats\")"
end
const USE_D64  = true
const D64_FROM = 7
"Returns Double64 for Nmax ≥ D64_FROM=7."
precision_for(Nmax) = (USE_D64 && HAVE_D64 && Nmax >= D64_FROM) ? Double64 : Float64

# Tolerances / grids (shared)
const OPTOL_GRID = [3e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7]
const ENTRY_TOLS = (1e-3, 1e-6, 1e-9)
const PREF_OPTOL = 1e-6
const TRIM_ATOL  = 1e-12
const GEN_DEPTH  = 20
const ETAS       = QUICK ? [0.5, 0.9648, 1.0] : [0.1, 0.3, 0.5, 0.765, 0.9648, 1.0, 1.2]
const ETA_DEPTH  = QUICK ? [0.1, 0.9648]       : [0.1, 0.3, 0.5, 0.765, 0.9648, 1.0]
maxelem_of(d)    = 3 * d^2 + 50

"
The ordinary Laguerre polynomial L_n(x) by three-term recurrence;
carrier_weight(n,η) = e^{−η²/2} L_n(η²) = ⟨n|D(η)|n⟩ is the grade-0 carrier weight d_n.
"
laguerre(n, x::T) where {T} = (n == 0 ? one(T) : n == 1 ? one(T) - x :
                  ((T(2n - 1) - x) * laguerre(n - 1, x) - T(n - 1) * laguerre(n - 2, x)) / T(n))
carrier_weight(n, eta::T) where {T} = exp(-eta^2 / 2) * laguerre(n, eta^2)   # d_n(eta)

"Generalised Laguerre L_n^{(k)}(x) by recurrence, in precision T."
function glag(n::Int, k::Int, x::T) where {T}
    n == 0 && return one(T)
    L0 = one(T); L1 = T(1 + k) - x
    for j in 1:n-1
        L0, L1 = L1, ((T(2j + 1 + k) - x) * L1 - T(j + k) * L0) / T(j + 1)
    end
    L1
end

"
Calculates the closed-form Cahill-Glauber element
⟨m|D(iη)|n⟩ = e^{−η²/2}(iη)^{|m-n|} √(min!/max!) L_{min}^{|m-n|}(η²)
This way we can use the exact analytical formula and we do not need to construct buffer state.
"
function disp_el(m::Int, n::Int, eta::RT, ::Type{CT}) where {RT,CT}
    x = eta^2; e = exp(-x / 2)
    if m >= n
        k = m - n; pref = sqrt(RT(factorial(big(n))) / RT(factorial(big(m))))
    else
        k = n - m; pref = sqrt(RT(factorial(big(m))) / RT(factorial(big(n))))
    end
    CT(pref * e * glag(min(m, n), k, x)) * (CT(0, eta))^k     # (i eta)^|m-n|
end


"Unit-normalise a generator by its 1-norm so op_rtol means the same across sizes."
unit_normalize(g) = (s = opnorm(Matrix(g), 1); s > 0 ? g / s : g)

"Lie closure of a list of (unit-normalised) Hermitian generators."
gen_algebra(generators, op_rtol; depth = GEN_DEPTH, maxelem) =
    generate_algebra_fast([sparse(unit_normalize(g)) for g in generators], depth;
                          op_rtol = op_rtol, trim_atol = TRIM_ATOL, maxelem = maxelem, verbose = 0)

"
The relative 1-norm of the least-squares remainder of target projected onto
span(mats). First it stack the vectorised basis as columns of V, then solves c = V vec(target),
 and returns ‖target − V c‖₁ / ‖target‖₁.
 "
function residual_relnorm(mats, target)
    isempty(mats) && return 1.0
    Tt = Matrix(target); n = size(Tt, 1)
    V  = reduce(hcat, [vec(Matrix(M)) for M in mats])
    t  = vec(Tt)
    c  = V \ t
    rem = reshape(t - V * c, n, n)
    Float64(opnorm(rem, 1) / opnorm(Tt, 1))
end

"""
Builds the residual-vs-depth curve.
For each commutator depth L, project target onto the basis elements generated within L brackets and record 
residual_relnorm. Then cliff depth = the L of the steepest single-step log-drop in the residual, while 
entry depths is the  first L at which the residual fall below each tol in tols.
"""
function target_depth(lie, target; blockmap = M -> Matrix(M), tols = ENTRY_TOLS)
    names = lie.elements_name
    els   = [blockmap(L) for L in lie.orthogonal_elements]
    maxL  = maximum(length.(names))
    curve = [residual_relnorm(els[findall(nm -> length(nm) <= L, names)], target) for L in 1:maxL]
    fl    = 1e-15
    drops = [log10(max(curve[L], fl)) - log10(max(curve[L+1], fl)) for L in 1:maxL-1]
    cliff = isempty(drops) ? 0 : argmax(drops)
    entry = map(tt -> (i = findfirst(L -> curve[L] < tt, 1:maxL); i === nothing ? -1 : i - 1), tols)
    (; cliff_depth = cliff, cliff_resid = curve[min(cliff + 1, maxL)], entry, curve, maxL)
end

"Pick the optimal operating tolerance: prefer PREF_OPTOL if it closes at d^2; else nearest stable."
function choose_operating(dims, d; pref = PREF_OPTOL)
    pick(set) = set[argmin([abs(log10(ot) - log10(pref)) for (ot, _) in set])]
    exact = [(ot, dm) for (ot, dm) in dims if dm == d^2]
    phys  = [(ot, dm) for (ot, dm) in dims if 0 < dm < d^2]
    if !isempty(exact)
        ot, dm = pick(exact); return ot, dm, "plateau dim==d^2"
    elseif !isempty(phys)
        ot, dm = pick(phys);  return ot, dm, "physical dim<d^2 (under-count)"
    else
        ot, dm = dims[argmin([dm - d^2 for (_, dm) in dims])]
        return ot, dm, "BORDERLINE over by $(dm - d^2)"
    end
end

"Calculates the number of grid tolerances giving exactly d^2, and their decade span."
function saturation_plateau(dims, d)
    sat = [ot for (ot, dm) in dims if dm == d^2]
    isempty(sat) && return (count = 0, decades = 0.0)
    (count = length(sat), decades = log10(maximum(sat)) - log10(minimum(sat)))
end


" 
Builds {N, K, σ_z, σ_x, K0} on the unbuffered register from the exact disp_el(d = 2(Nmax+1));
K0 is the carrier (grade-0) part of the drive. Returns sparse generators plus the dense Nfull 
for the g0 study.
"
function register_ops(Nmax, eta; prec::Type{RT} = Float64) where {RT}
    CT = Complex{RT}; Nr = Nmax + 1
    Dr = CT[disp_el(m, n, RT(eta), CT) for m in 0:Nmax, n in 0:Nmax]
    p10 = CT[0 0; 1 0]; p01 = CT[0 1; 0 0]
    sx = CT[0 1; 1 0]; sz = CT[1 0; 0 -1]; I2 = Matrix{CT}(I, 2, 2)
    Ireg = Matrix{CT}(I, Nr, Nr); numr = diagm(0 => CT.(0:Nmax))
    K  = kron(p10, Dr) + kron(p01, Dr')
    Nd = kron(I2, numr); Sz = kron(sz, Ireg); Sx = kron(sx, Ireg)
    K0 = kron(sx, diagm(0 => CT[CT(carrier_weight(n, RT(eta))) for n in 0:Nmax]))
    (; N = sparse(Nd), K = sparse(K), Sz = sparse(Sz), Sx = sparse(Sx), K0 = sparse(K0),
       Nfull = Nd, d = 2Nr, Nmax, eta)
end

" {A,B} generators, target, register block-map for Study 2. "
function make_ops(Nmax, eta; prec::Type{RT} = Float64) where {RT}
    dreg = 2 * (Nmax + 1)
    CT = Complex{RT}; Nr = Nmax + 1
    Dr  = CT[disp_el(m, n, RT(eta), CT) for m in 0:Nmax, n in 0:Nmax]
    p10 = CT[0 0; 1 0]; p01 = CT[0 1; 0 0]
    sx  = CT[0 1; 1 0]; sz = CT[1 0; 0 -1]; I2 = Matrix{CT}(I, 2, 2)
    Ireg = Matrix{CT}(I, Nr, Nr); numr = diagm(0 => CT.(0:Nmax))
    K  = kron(p10, Dr) + kron(p01, Dr')
    N  = kron(I2, numr)
    A  = RT(OMEGA_REL) * N + RT(0.5) * K     # == build_system's A = w N + 1/2 K
    B  = kron(sz, Ireg)
    target = kron(sx, Ireg)
    return (; A = sparse(A), B = sparse(B), target = target, blockmap = L -> Matrix(L),
              d = dreg, d2 = dreg^2, dbuild = dreg, dsim = dreg, buffer = 0)
end

"
For each Nmax at the Sr-88 η, sweeps op_rtol over OPTOL_GRID: generates the algebra,
records dim, dim/d², over-saturation flag, cliff depth, entry depths, and the full
residual curve; then choose_operating picks the operating tolerance per Nmax. Writes
reach_optol_sweep/curves/operating.csv. Returns the operating-tolerance map.
"
function tri_block_sweep()
    println("[1.0] op_rtol sweep over all Nmax (eta=Sr88)")
    sp = joinpath(DATA, "reach_optol_sweep.csv"); cp = joinpath(DATA, "reach_optol_curves.csv")
    opf = joinpath(DATA, "reach_operating.csv")
    open(sp, "w") do io; println(io, "Nmax,eta,d,d2,op_rtol,dim,dim_over_d2,oversaturated,cliff_depth,entry_1e3,entry_1e6,entry_1e9"); end
    open(cp, "w") do io; println(io, "Nmax,op_rtol,depth,residual"); end
    open(opf, "w") do io; println(io, "Nmax,d,d2,op_rtol_operating,plateau_dim,note"); end
    operating = Dict{Int,Float64}()
    for Nmax in NMAXS
        eta = ETA_SR88; ops = register_ops(Nmax, eta; prec = precision_for(Nmax))
        @printf("  Nmax=%d (d=%d, d^2=%d, %s):\n", Nmax, ops.d, ops.d^2,
                precision_for(Nmax) == Float64 ? "F64" : "D64")
        @printf("    %-9s %-6s %-8s %-6s %-6s %-12s\n", "op_rtol", "dim", "dim/d^2", "over?", "cliff", "entry(1e-6)")
        dims = Tuple{Float64,Int}[]
        for ot in OPTOL_GRID
            lie = gen_algebra([ops.N, ops.K, ops.Sz], ot; maxelem = maxelem_of(ops.d))
            td  = target_depth(lie, ops.Sx); over = lie.dim > ops.d^2
            push!(dims, (ot, lie.dim))
            @printf("    %-9.0e %-6d %-8.3f %-6s %-6d %-12d\n",
                    ot, lie.dim, lie.dim / ops.d^2, over ? "OVER" : "-", td.cliff_depth, td.entry[2])
            open(sp, "a") do io
                println(io, @sprintf("%d,%.4f,%d,%d,%.0e,%d,%.4f,%d,%d,%d,%d,%d",
                        Nmax, eta, ops.d, ops.d^2, ot, lie.dim, lie.dim / ops.d^2, over,
                        td.cliff_depth, td.entry[1], td.entry[2], td.entry[3]))
            end
            for (L, r) in enumerate(td.curve)
                open(cp, "a") do io; println(io, @sprintf("%d,%.0e,%d,%.6e", Nmax, ot, L-1, r)); end
            end
        end
        ot, dm, note = choose_operating(dims, ops.d)
        operating[Nmax] = ot
        @printf("    -> operating op_rtol=%.0e, plateau dim=%d/%d  [%s]\n", ot, dm, ops.d^2, note)
        open(opf, "a") do io; println(io, @sprintf("%d,%d,%d,%.0e,%d,%s", Nmax, ops.d, ops.d^2, ot, dm, note)); end
    end
    println("  wrote ", basename(sp), ", ", basename(cp), ", ", basename(opf))
    operating
end

"
At each (Nmax, η) (η grid), records dim/d² and cliff depth at the operating tolerance, 
the saturation-vs-η picture. Writes reach_saturation.csv.
"
function tri_block_saturation(operating)
    println("[1.A] Plateau dim vs d^2 vs eta")
    path = joinpath(DATA, "reach_saturation.csv")
    open(path, "w") do io; println(io, "Nmax,eta,d,d2,dim,dim_over_d2,oversaturated,cliff_depth"); end
    for Nmax in NMAXS, eta in ETAS
        ops = register_ops(Nmax, eta; prec = precision_for(Nmax)); ot = operating[Nmax]
        lie = gen_algebra([ops.N, ops.K, ops.Sz], ot; maxelem = maxelem_of(ops.d))
        td  = target_depth(lie, ops.Sx); over = lie.dim > ops.d^2
        @printf("  Nmax=%d eta=%.4f: dim=%d/%d (%.3f)%s cliff=%d\n",
                Nmax, eta, lie.dim, ops.d^2, lie.dim / ops.d^2, over ? " OVER" : "", td.cliff_depth)
        open(path, "a") do io
            println(io, @sprintf("%d,%.4f,%d,%d,%d,%.4f,%d,%d",
                    Nmax, eta, ops.d, ops.d^2, lie.dim, lie.dim / ops.d^2, over, td.cliff_depth))
        end
    end
    println("  wrote ", basename(path))
end

"
At a fixed register depth, records the residual-vs-depth curve across η. Writes reach_depth_curve.csv.
"
function tri_block_depth_curve(operating)
    println(" [1.B] Residual-vs-depth vs eta (Nmax=$(NMAX_DEPTH))")
    path = joinpath(DATA, "reach_depth_curve.csv")
    open(path, "w") do io; println(io, "eta,depth,residual"); end
    ot = operating[NMAX_DEPTH]
    for eta in ETA_DEPTH
        ops = register_ops(NMAX_DEPTH, eta; prec = precision_for(NMAX_DEPTH))
        lie = gen_algebra([ops.N, ops.K, ops.Sz], ot; maxelem = maxelem_of(ops.d))
        td  = target_depth(lie, ops.Sx)
        @printf("  eta=%.4f: cliff=%d, entry@1e-6=%d (op_rtol=%.0e)\n", eta, td.cliff_depth, td.entry[2], ot)
        for (L, r) in enumerate(td.curve)
            open(path, "a") do io; println(io, @sprintf("%.4f,%d,%.6e", eta, L-1, r)); end
        end
    end
    println("  wrote ", basename(path))
end


"
Builds a basis of the recoil-free centraliser g0 ∩ Herm ≅ ⊕_n u(2)_n ({I₂,σ_x,σ_y,σ_z}⊗|n⟩⟨n|), 
checks ‖[X,N]‖₁ ≈ 0 for each (grade-0 verification), measures dim(g0) by the numerical rank of 
the stacked vectorised basis (atol=1e-8) against the expected 4(Nmax+1), and reports the max 
residual of every g0 element projected onto the generated algebra. Writes reach_g0_structure.csv.
"
function tri_block_g0_structure(operating)
    println("[1.D] Structure of the recoil-free subspace g0 ")
    path = joinpath(DATA, "reach_g0_structure.csv")
    open(path, "w") do io; println(io, "Nmax,eta,dim_g0,expected_4Np1,grade0_maxerr,reach_maxresid,oversaturated"); end
    for Nmax in NMAXS
        eta = ETA_SR88; RT = precision_for(Nmax); CT = Complex{RT}
        ops = register_ops(Nmax, eta; prec = RT); N = Matrix(ops.Nfull); ot = operating[Nmax]
        sx = CT[0 1; 1 0]; sy = CT[0 -im; im 0]; sz = CT[1 0; 0 -1]; I2 = Matrix{CT}(I, 2, 2)
        g0 = SparseMatrixCSC{CT,Int}[]
        for n in 0:Nmax
            e = zeros(CT, Nmax+1, Nmax+1); e[n+1, n+1] = 1
            for s in (I2, sx, sy, sz); push!(g0, sparse(kron(s, e))); end
        end
        grade0_err = Float64(maximum(opnorm(Matrix(X * N - N * X), 1) for X in g0))               # [X,N]=0 ?
        dimg0 = rank(hcat([vec(ComplexF64.(Matrix(X))) for X in g0]...); atol = 1e-8)
        lie = gen_algebra([ops.N, ops.K, ops.Sz], ot; maxelem = maxelem_of(ops.d))
        over = lie.dim > ops.d^2
        reach = over ? NaN : Float64(maximum(residual_relnorm(lie.orthogonal_elements, X) for X in g0))
        @printf("  Nmax=%d: dim(g0)=%d (expect %d), grade0=%.1e, reach=%s%s\n",
                Nmax, dimg0, 4*(Nmax+1), grade0_err,
                over ? "n/a" : @sprintf("%.1e", reach), over ? " [over-sat]" : "")
        open(path, "a") do io
            println(io, @sprintf("%d,%.4f,%d,%d,%.6e,%.6e,%d",
                    Nmax, eta, dimg0, 4*(Nmax+1), grade0_err, isnan(reach) ? -1.0 : reach, over))
        end
    end
    println("  wrote ", basename(path))
end

"
Tabulates carrier weights d_n(η), sweeps the carrier-algebra dimension {N, σ_z, K0}
against the expected 3Nmax+4, and reports the minimum |d_n| and minimum |d_i+d_j|.
Writes reach_carrier*.csv.
"
function tri_block_carrier()
    println("[1.E] Carrier weights + carrier algebra G0 ")
    wpath = joinpath(DATA, "reach_carrier_weights.csv")
    open(wpath, "w") do io; println(io, "eta,n,weight"); end
    for eta in ETA_DEPTH, n in 0:10
        open(wpath, "a") do io; println(io, @sprintf("%.4f,%d,%.6e", eta, n, Float64(carrier_weight(n, eta)))); end
    end
    swp = joinpath(DATA, "reach_carrier_sweep.csv"); path = joinpath(DATA, "reach_carrier.csv")
    open(swp, "w") do io; println(io, "Nmax,op_rtol,dim_G0"); end
    open(path, "w") do io; println(io, "Nmax,eta,dim_G0,expected_3Np4,op_rtol,min_abs_dn,min_abs_sum,carrier_ok"); end
    @printf("  %-5s %-8s %-8s %-12s %-9s\n", "Nmax", "dim G0", "3N+4", "op_rtol", "min|dn|")
    for Nmax in NMAXS
        eta = ETA_SR88; ops = register_ops(Nmax, eta; prec = precision_for(Nmax))
        dims = Tuple{Float64,Int}[]
        for ot in OPTOL_GRID
            g = gen_algebra([ops.N, ops.Sz, ops.K0], ot; maxelem = maxelem_of(ops.d))
            push!(dims, (ot, g.dim))
            open(swp, "a") do io; println(io, @sprintf("%d,%.0e,%d", Nmax, ot, g.dim)); end
        end
        ot, dm, _ = choose_operating(dims, ops.d)
        dn = [Float64(carrier_weight(n, eta)) for n in 0:Nmax]
        min_abs = minimum(abs.(dn))
        min_sum = minimum([abs(dn[i] + dn[j]) for i in 1:length(dn) for j in i+1:length(dn)]; init = Inf)
        ok = (min_abs > 1e-6) && (min_sum > 1e-6)
        @printf("  %-5d %-8d %-8d %-12.0e %-9.3e\n", Nmax, dm, 3*Nmax+4, ot, min_abs)
        open(path, "a") do io
            println(io, @sprintf("%d,%.4f,%d,%d,%.0e,%.6e,%.6e,%d",
                    Nmax, eta, dm, 3*Nmax+4, ot, min_abs, min_sum, ok))
        end
    end
    println("  wrote ", basename(path), ", ", basename(swp), ", ", basename(wpath))
end

function study_tri()
    println("Study 1 three-generator {N, K, sigma_z} (upper bound)")
    operating = tri_block_sweep()
    tri_block_saturation(operating)
    tri_block_depth_curve(operating)
    tri_block_g0_structure(operating)
    tri_block_carrier()
end


" Fresh headers once per run (so re-runs do not duplicate rows); blocks then append. "
function ab_init_csvs()
    heads = Pair{String,String}[
        "reach_AB_optol_sweep.csv" => "Nmax,eta,buffer,dsim,d,d2,op_rtol,dim,dim_over_d2,oversaturated,cliff_depth,entry_1e3,entry_1e6,entry_1e9",
        "reach_AB_optol_curves.csv" => "Nmax,op_rtol,depth,residual",
        "reach_AB_operating.csv" => "Nmax,d,d2,op_rtol_operating,plateau_dim,note",
        "reach_AB_certificate.csv" => "Nmax,d,d2,op_rtol_operating,plateau_dim,plateau_count,plateau_decades,saturated",
        "reach_AB_saturation.csv" => "Nmax,eta,buffer,dsim,d,d2,dim,dim_over_d2,oversaturated,cliff_depth",
        "reach_AB_depth_curve.csv" => "eta,depth,residual",
    ]
    for (f, h) in heads
        open(io -> println(io, h), joinpath(DATA, f), "w")
    end
end

"
The {A,B} analogue of tri_block_sweep, on the build space dbuild via make_ops. Sweeps
op_rtol, records dimension/ratio/over-flag/cliff/entry and residual curves, then
choose_operating + saturation_plateau produce the operating tolerance and the
saturation certificate (saturated = dim==dbuild² and plateau ≥ 2 points). Writes
reach_AB_optol_sweep/curves/operating/certificate.csv. Returns the operating tolerance
"
function ab_block_sweep(Nmax)
    sp = joinpath(DATA, "reach_AB_optol_sweep.csv"); cp = joinpath(DATA, "reach_AB_optol_curves.csv")
    opf = joinpath(DATA, "reach_AB_operating.csv");  cert = joinpath(DATA, "reach_AB_certificate.csv")
    eta = ETA_SR88
    ops = make_ops(Nmax, eta; prec = precision_for(Nmax))
    @printf("  Nmax=%d (d=%d, d^2=%d, dbuild=%d, %s):\n", Nmax, ops.d, ops.d2, ops.dbuild,
            precision_for(Nmax) == Float64 ? "F64" : "D64")
    @printf("    %-9s %-6s %-9s %-6s %-6s %-12s\n", "op_rtol", "dim", "dim/db^2", "over?", "cliff", "entry(1e-6)")
    dims = Tuple{Float64,Int}[]
    for ot in OPTOL_GRID
        lie = gen_algebra([ops.A, ops.B], ot; maxelem = maxelem_of(ops.dbuild))
        td  = target_depth(lie, ops.target; blockmap = ops.blockmap)
        over = lie.dim > ops.dbuild^2
        ratio = lie.dim / ops.dbuild^2                          # ratio on the BUILD space
        push!(dims, (ot, lie.dim))
        @printf("    %-9.0e %-6d %-9.3f %-6s %-6d %-12d\n",
                ot, lie.dim, ratio, over ? "OVER" : "-", td.cliff_depth, td.entry[2])
        open(sp, "a") do io
            println(io, @sprintf("%d,%.4f,%d,%d,%d,%d,%.0e,%d,%.4f,%d,%d,%d,%d,%d",
                    Nmax, eta, ops.buffer, ops.dsim, ops.d, ops.d2, ot, lie.dim,
                    ratio, over, td.cliff_depth, td.entry[1], td.entry[2], td.entry[3]))
        end
        for (L, r) in enumerate(td.curve)
            open(cp, "a") do io; println(io, @sprintf("%d,%.0e,%d,%.6e", Nmax, ot, L - 1, r)); end
        end
    end
    ot, dm, note = choose_operating(dims, ops.dbuild)
    pl = saturation_plateau(dims, ops.dbuild)
    sat = (dm == ops.dbuild^2) && (pl.count >= 2)
    @printf("    -> operating op_rtol=%.0e, plateau dim=%d/%d  [%s]  (plateau: %d pts, %.1f decades)\n",
            ot, dm, ops.dbuild^2, note, pl.count, pl.decades)
    open(opf, "a") do io; println(io, @sprintf("%d,%d,%d,%.0e,%d,%s", Nmax, ops.d, ops.d2, ot, dm, note)); end
    open(cert, "a") do io
        println(io, @sprintf("%d,%d,%d,%.0e,%d,%d,%.2f,%d",
                Nmax, ops.dbuild, ops.dbuild^2, ot, dm, pl.count, pl.decades, sat))
    end
    ot
end

"
{A,B} saturation across η at the operating tolerance. Writes reach_AB_saturation.csv.
"
function ab_block_saturation(Nmax, ot)
    path = joinpath(DATA, "reach_AB_saturation.csv")
    for eta in ETAS
        ops = make_ops(Nmax, eta; prec = precision_for(Nmax))
        lie = gen_algebra([ops.A, ops.B], ot; maxelem = maxelem_of(ops.dbuild))
        td  = target_depth(lie, ops.target; blockmap = ops.blockmap); over = lie.dim > ops.dbuild^2
        ratio = lie.dim / ops.dbuild^2
        @printf("  Nmax=%d eta=%.4f: dim=%d/%d (%.3f)%s cliff=%d\n",
                Nmax, eta, lie.dim, ops.dbuild^2, ratio, over ? " OVER" : "", td.cliff_depth)
        open(path, "a") do io
            println(io, @sprintf("%d,%.4f,%d,%d,%d,%d,%d,%.4f,%d,%d",
                    Nmax, eta, ops.buffer, ops.dsim, ops.d, ops.d2, lie.dim, ratio, over, td.cliff_depth))
        end
    end
end

"
{A,B} residual-vs-depth across η at one representative register. Writes reach_AB_depth_curve.csv.
"
function ab_block_depth_curve(Nmax, ot)
    path = joinpath(DATA, "reach_AB_depth_curve.csv")
    for eta in ETA_DEPTH
        ops = make_ops(Nmax, eta; prec = precision_for(Nmax))
        lie = gen_algebra([ops.A, ops.B], ot; maxelem = maxelem_of(ops.dbuild))
        td  = target_depth(lie, ops.target; blockmap = ops.blockmap)
        @printf("  eta=%.4f: cliff=%d, entry@1e-6=%d (op_rtol=%.0e)\n", eta, td.cliff_depth, td.entry[2], ot)
        for (L, r) in enumerate(td.curve)
            open(path, "a") do io; println(io, @sprintf("%.4f,%d,%.6e", eta, L - 1, r)); end
        end
    end
end

"
Initialises CSVs, loops Nmax (each in a try/catch so one failure does not abort the rest, since especially
the Double64 studies can take very long and the virtual system was crashing), running the sweep, 
the η-saturation; finally the depth-vs-η curve at one 
representative register.
"
function study_ab()
    println("Study 2 two-generator {A, B} (physically reachable) ")
    ab_init_csvs()
    ab_operating = Dict{Int,Float64}()
    for Nmax in NMAXS
        try
            println("\n  -- {A,B} Nmax=$(Nmax) --")
            ot = ab_block_sweep(Nmax)
            println("     saturation vs eta:")
            ab_block_saturation(Nmax, ot)
            ab_operating[Nmax] = ot
        catch err
            @warn "AB Nmax=$(Nmax) failed; continuing with the rest." exception = (err, catch_backtrace())
        end
    end
    # depth-vs-eta at ONE representative register (single-Nmax schema, like Study 1)
    if haskey(ab_operating, NMAX_DEPTH)
        println("\n  -- {A,B} residual vs depth vs eta (Nmax=$(NMAX_DEPTH)) --")
        ab_block_depth_curve(NMAX_DEPTH, ab_operating[NMAX_DEPTH])
    end
    println("  Study 2 done.")
end


function main()
    println("UNIFIED Lie-algebra reachability via MagnusTensor (arXiv:2512.20357)")
    println("eta_Sr88=$(ETA_SR88), omega/Omega=$(OMEGA_REL); closed-form exact D; UNIT-NORMALISED 1-norm")
    println("Nmax=$(NMAXS)  (override via ARGS: `2 3 4`, `2:7`, `2,3,4`);  depth-curve register=$(NMAX_DEPTH)")
    @printf("trim_atol=%.0e, pref op_rtol=%.0e, depth<=%d; Double64 for Nmax>=%d (have D64: %s); QUICK=%s\n",
            TRIM_ATOL, PREF_OPTOL, GEN_DEPTH, D64_FROM, HAVE_D64, QUICK)
    RUN_TRI && study_tri()
    RUN_AB  && study_ab()
    println("\nAll done. CSVs in ", DATA, " ; render with scripts/visualize_reachability.jl")
end

main()