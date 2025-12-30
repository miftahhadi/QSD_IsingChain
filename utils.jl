using LinearAlgebra, SparseArrays
using FastExpm

kronDelta(i,j) = (i==j) ? 1.0 : 0.0

# Build the BdG Hamiltonian
function buildHbdG(J, L)
    diag1 = ones(L-1)
    A = spdiagm(1 => ComplexF64(J)*diag1, -1 => ComplexF64(J)*diag1)
    B = spdiagm(1 => ComplexF64(J)*diag1, -1 => -ComplexF64(J)*diag1)
    
    return 0.5 * [-A' -B; B A]
end

# Initial correlation matrix G(0)
#   [ <c† c>    <c† c†> ;
#     <c c>     <c c†> ]
# and the initial state is |000...0>
function GInit(L)
    zeroL = zeros(ComplexF64, L, L)
    idtyL = Matrix{ComplexF64}(I, L, L)
    top = hcat(zeroL, zeroL)
    bot = hcat(zeroL, idtyL)
    return vcat(top, bot)
end

# Build the "diag_block" used to calculate G(t) = U * diag_block * U†
function diagBlockMatrix(L)
    top = hcat(Matrix{ComplexF64}(I, L, L), zeros(ComplexF64, L, L))
    bot = hcat(zeros(ComplexF64, L, L), zeros(ComplexF64, L, L))

    return vcat(top, bot)
end

# Construct U0 from G0
# To verify just do U0 * diag_block * U0† and see if you get G0
function buildU0(L)
    U0 = [zeros(ComplexF64, L, L) Matrix{ComplexF64}(I, L, L);   
        Matrix{ComplexF64}(I, L, L) zeros(ComplexF64, L, L)]

    return U0
end

# Majorana transform
function VMajorana(L)
    idtyL = Matrix{ComplexF64}(I, L, L)
    top = hcat(idtyL, idtyL)
    bot = hcat(1im * idtyL, -1im * idtyL)

    return vcat(top, bot)
end

function getWTilde(G, idty2L, VMaj)
    # Get the Majorana correlation matrix M
    M = VMaj * G * VMaj'

    # Use M = (I + i Wtilde) --> Wtilde = -i(M - I)
    Wtilde = 1im * (idty2L - M)
    
    return Wtilde
end

function getMajoranaIndices(L, Lsub::Int64)
    # Majorana indices for sites 1..Lsub 
    # There are 2 Majorana modes per site
    maj_indices = Vector{Int}(undef, 2*Lsub)
    for j in 1:Lsub
        maj_indices[j] = j
        maj_indices[Lsub + j] = L + j
    end
    
    return maj_indices
end

function getEntanglementEntropy(G, idty2L, VMajorana, majorana_indices)
    Wtilde = getWTilde(G, idty2L, VMajorana)
    
    # Sanity check 
    if any(isnan, Wtilde) || any(isinf, Wtilde)
        println("NaN/Inf detected in Wtilde — skipping entropy calculation")
        return 0.0
    end

    # Restrict Wtilde to the subsystem
    Wtilde = Wtilde[majorana_indices, majorana_indices]

    # Handling the numerical drift
    Wtilde .= real.(Wtilde) # Force to be real
    Wtilde .= 0.5 .* (Wtilde .- Wtilde') # Force to be antisymmetric
    
    # Eigenvalues of Wtilde
    λ_all = try
            real.(eigvals(im * Wtilde))        
    catch e 
        println("eigvals failed to converge: $e — skipping entropy calculation")
        return 0.0
    end

    λ_k = filter(x -> x > 0.0, λ_all) # Keep only positive eigenvalues

    # Entanglement entropy 
    S_ent = 0.0

    for λ in λ_k
        p = (1 + λ) / 2
        q = (1 - λ) / 2
        if p > 0; S_ent -= p * log(p); end
        if q > 0; S_ent -= q * log(q); end
    end

    return S_ent
end

# Prepare computation for QSD
function prepare(J, L, Lsub, dt)
    # Build the H_BdG
    Ising_HBdG = buildHbdG(J, L)

    # Compute the exponentiated H_BdG
    exp_H = fastExpm(2 * 1im * Ising_HBdG * dt)
    
    # diag_block matrix for G(t) = U * diag_block * U†
    diag_block = diagBlockMatrix(L)

    # Majorana indices
    maj_indices = getMajoranaIndices(L, Lsub)

    # Majorana transformation matrix
    VMaj = VMajorana(L)

    idty2L = Matrix{ComplexF64}(I, 2*L, 2*L)

    return exp_H, diag_block, VMaj, idty2L, maj_indices

end

function isComputeS(step, dt)
    # How many steps to make it 1.0 second?
    ref_step = Int(round(1.0 / dt))

    # What is the current time?
    current_time = (step-1) * dt

    res = true
    if current_time > 1.0 && step % ref_step != 0
        res = false
    end
    return res
end