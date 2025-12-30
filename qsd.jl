using LinearAlgebra, Random, Statistics, SparseArrays
using Base.Threads, Distributed

include("utils.jl")

# Main QSD function
function runQSD(
    L::Int,
    exp_H,
    diag_block,
    VMaj,
    idty2L,
    majorana_indices,
    gamma::Float64,
    t_max::Float64,
    dt::Float64,
    itraj=nothing,
    rng=Random.GLOBAL_RNG
)
    # Prepares the parameters
    nsteps = round(Int, t_max / dt)
    # We will compute S_ent at each step up to this threshold
    threshold_t = 2.5 # in seconds
    # Size of the result containers 
    res_size = getResultArraySize(nsteps, dt, threshold_t)

    # Prepare G0 and U0
    G = GInit(L)
    U = buildU0(L)

    # Preallocate containers. 
    times = zeros(Float64, res_size+1)
    entropy_arr = zeros(Float64, res_size+1)
    dξ = zeros(Float64, L)
    n_i = zeros(Float64, L)

    # Temporary containers for computations
    temp1 = similar(U) # can hold exp_H * U0
    temp2 = similar(U) # can hold exp_dT * temp1

    dT_vec = zeros(Float64, L)
    exp_dTplus = zeros(Float64, L)
    exp_dTminus = zeros(Float64, L)

    # Compute entanglement entropy at t=0
    # Of course, should be zero
    S_ent = getEntanglementEntropy(G, idty2L, VMaj, majorana_indices)

    # Store the results
    times[1] = 0
    entropy_arr[1] = S_ent

    # Let's start the time evolution
    start_step = 2
    for step in start_step:(nsteps+1)
        # What is current time?
        current_time = (step-1) * dt
        # Do we compute the entanglement entropy at this step?
        compute_Sent = isComputeS(step, dt)

        # Sample Wiener increment
        # dξ = sqrt(gamma * dt) * randn(rng, L)
        randn!(rng, dξ) # dξ ~ N(0,1)
        @inbounds for i in 1:L
            dξ[i] *= sqrt(gamma * dt) 
        end

        # Get the occupation number <n_i> = <c_i† c_i> from G
        # In G, it is block G[1:L, 1:L]
        # We use G from the previous step
        @inbounds for i in 1:L
            n_i[i] = real(G[i, i])
        end

        # Noise matrix
        @inbounds for i in 1:L 
            dT_vec[i] = 0.5 * (dξ[i] + gamma*dt * (2*n_i[i] - 1.0))
        end         

        @inbounds for i in 1:L
            v = 2.0 * dT_vec[i]
            exp_dTplus[i] = exp(v)
            exp_dTminus[i] = exp(-v)
        end

        # Begin calculating U_tilde
        mul!(temp1, exp_H, U) # temp1 = exp_H * U

        @inbounds for j in 1:L 
            @views temp2[j, :]      .= exp_dTminus[j] * temp1[j, :]
            @views temp2[L + j, :]  .= exp_dTplus[j] * temp1[L + j, :]
        end

        Utilde = temp2

        # QR decomposition
        Q, R = qr(Utilde)
        U .= Matrix(Q)

        # Compute new G for current time 
        mul!(temp1, U, diag_block)  # temp1 = U * diag_block
        mul!(G, temp1, U')          # G = temp1 * U' 

        # Enforce the hermiticity of G due to numerical errors
        adjoint!(temp2, G)  # temp2 = G†

        G .+= temp2     # G = G + G†
        G .*= 0.5       # G = 0.5 * (G + G†)

        if compute_Sent
            # Compute the entanglement entropy
            S_ent = getEntanglementEntropy(G, idty2L, VMaj, majorana_indices)

            # Store the results
            times[step] = current_time
            entropy_arr[step] = S_ent

            if itraj !== nothing
                println("Trajectory #$(itraj)")                
            end
            println("Time: $(current_time) / $(t_max)\nS_ent = $(S_ent)\n")
        end
    end

    if itraj !==nothing
        println("Trajectory #$(itraj) completed at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    end
    
    return times, entropy_arr    

end

# Multi-threaded calculation of QSD ensemble
function runQSDEnsembleParallel(
    L::Int,
    ntraj::Int,
    exp_H,
    diag_block,
    VMaj,
    idty2L,
    majorana_indices,
    gamma::Float64,
    t_max::Float64,
    dt::Float64,
    base_seed::Int=1234
)
    nsteps = round(Int, t_max / dt)

    # Containers
    S_trajs = zeros(Float64, nsteps+1, ntraj)
    times = collect(0:nsteps) .* dt

    @threads for j in 1:ntraj 
        seed = base_seed + j
        rng = Xoshiro(seed)
        t, S = runQSD(
            L,
            exp_H,
            diag_block,
            VMaj,
            idty2L,
            majorana_indices,
            gamma,
            t_max,
            dt,
            j,
            rng
        )

        S_trajs[:, j] = S
    end

    mean_S = mean(S_trajs, dims=2)[:]
    std_S = std(S_trajs, dims=2)[:]

    return times, mean_S, std_S, S_trajs
end

# Multi-processing calculation of QSD ensemble
# Single trajectory function for distributed computing
function single_traj(
    j, L, exp_H, diag_block, VMaj, idty2L, majorana_indices, gamma, t_max, dt, base_seed
)
    try
        println("Trajectory #$j is running on process $(myid())")
        rng = Xoshiro(base_seed + j)
        t, S = runQSD(L, exp_H, diag_block, VMaj, idty2L, majorana_indices, gamma, t_max, dt, j, rng)
        return S
    catch e 
        @warn "Trajectory #$(j) failed on worker $(myid()) with error: $e"
        return fill(NaN, round(Int, t_max / dt)+1)
    end
end

function runQSDEnsembleDistributed(
    L::Int,
    ntraj::Int,
    exp_H,
    diag_block,
    VMaj,
    idty2L,
    majorana_indices,
    gamma::Float64,
    t_max::Float64,
    dt::Float64,
    base_seed::Int=1234
)
    nsteps = round(Int, t_max / dt)
    times = collect(0:nsteps) .* dt

    # Run each trajectory on a different worker
    S_list = pmap(j -> single_traj(
            j, L, exp_H, diag_block, VMaj, idty2L, majorana_indices, gamma, t_max, dt, base_seed
        ), 1:ntraj)

    # Collect into an array (each column = one trajectory)
    S_trajs = hcat(S_list...)  # size: (nsteps+1, ntraj)
    mean_S = mean(S_trajs, dims=2)[:]
    std_S = std(S_trajs, dims=2)[:]

    return times, mean_S, std_S, S_trajs
end