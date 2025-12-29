# Quantum State Diffusion (QSD) simulation main script
using JLD2

using Distributed
using SlurmClusterManager

# ===========================================
#  SLURM-AWARE DISTRIBUTED SETUP
# ===========================================

if haskey(ENV, "SLURM_JOB_ID")
    try
        n = parse(Int, ENV["SLURM_NTASKS"])
        addprocs(SlurmManager())
        @info "Running under SLURM: added $(nworkers()) workers"
    catch e
        @warn "Could not add SLURM workers: $e"
    end
else
    @info "Running locally (no SLURM detected)"
end

# ===========================================
#  Load project code
# ===========================================
@everywhere begin
    using Random
    using Dates

    include("params.jl")
    include("qsd.jl")
end

#=============================#
#      Parse CLI argument     #
#=============================#
if length(ARGS) < 2
    println("Usage: julia main.jl <gamma> <L_chain>")
    println("OR FOR ENSEMBLE: julia main.jl <gamma> <L_chain> <N_traj>")
    exit(1)
end

γ = parse(Float64, ARGS[1])
L_chain = parse(Int, ARGS[2])
parallel = false

if length(ARGS) == 3
    parallel = true
    N_traj = parse(Int, ARGS[3])    
end

#===================#
#      Logging      #
#===================#
time_start = now()
timestamp = Dates.format(now(), "yyyymmdd_HH-MM-SS")

if parallel
    filename = "qsd_J$(J)_γ$(γ)_L$(L_chain)_N_traj$(N_traj)_$(timestamp).jld2"
else
    filename = "qsd_J$(J)_γ$(γ)_L$(L_chain)_single_$(timestamp).jld2"
end

# Set log file and save file paths
logfile = joinpath(outdir, "log_" * replace(filename, ".jld2" => ".txt"))
savefile = joinpath(outdir, filename)

println("Start time: " * Dates.format(time_start, "yyyy-mm-dd HH:MM:SS"))
println("Parameters: \nJ=$(J), \nγ=$(γ), \nL_chain=$(L_chain), \nt_max=$(t_max), \ndt=$(dt), \nsteps=$(round(Int, t_max / dt) + 1)")

if parallel
    println("Running ensemble with N_traj=$(N_traj)")
else
    println("Running single trajectory")
end

println("Results will be saved to: " * savefile)

#=============================#
#      Prepare Run Models     #
#=============================#
function runSingle(exp_H, diag_block, VMaj, idty2L, majorana_indices)
    times, S_ent = runQSD(
        L_chain,
        exp_H,
        diag_block,
        VMaj,
        idty2L,
        majorana_indices,
        γ,
        t_max,
        dt
    )
    
    jldsave(savefile, times=times, S_ent=S_ent)
    return
end

function runParallel(exp_H, diag_block, VMaj, idty2L, majorana_indices)
    @info "Running in parallel with $(nworkers()) workers..."

    if nworkers() > 1
        @info "Using distributed ensembles..."
        times, mean_S, std_S, S_trajs = runQSDEnsembleDistributed(
            L_chain,
            N_traj,
            exp_H,
            diag_block,
            VMaj,
            idty2L,
            majorana_indices,
            γ,
            t_max,
            dt
        )
    else
        @info "Running locally with multi-threading..."
        times, mean_S, std_S, S_trajs = runQSDEnsembleParallel(
            L_chain,
            N_traj,
            exp_H,
            diag_block,
            VMaj,
            idty2L,
            majorana_indices,
            γ,
            t_max,
            dt
        )
    end

    jldsave(savefile, times=times, mean_S=mean_S, std_S=std_S, S_trajs=S_trajs)
    return
end

#=============================#
#        Execute Run          #
#=============================#
@info "Preparing simulation..."
Lsub = div(L_chain, 4)
exp_H, diag_block, VMaj, idty2L, majorana_indices = prepare(J, L_chain, Lsub, dt)

if parallel
    runParallel(exp_H, diag_block, VMaj, idty2L, majorana_indices)
else
    runSingle(exp_H, diag_block, VMaj, idty2L, majorana_indices)
end

println("Simulation completed. Results saved to: " * savefile)

time_end = now()
elapsed_time = Dates.value(time_end - time_start) / 1000  # in seconds

println("End time: " * Dates.format(time_end, "yyyy-mm-dd HH:MM:SS"))
println("Execution time: $(elapsed_time/60) minutes ($(elapsed_time/3600) hours)")
println("Log saved to " * logfile)