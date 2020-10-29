using SparseArrays, LinearAlgebra

using LinearAlgebra: givensAlgorithm

function linear_part_sparse(n, α)
    h = 1.0 / n

    # Create the linear part as a sparse matrix.
    is = Vector{Int}()
    js = Vector{Int}()
    vs = Vector{Float64}()
    
    # Save indices
    indices = reshape(1:(n - 1)^2, n - 1, n - 1)

    for x = 1 : n - 1, y = 1 : n - 1
        I = indices[x, y]

        # central
        push!(is, I)
        push!(js, I)
        push!(vs, -4.0 / h^2)

        # north
        if y < n - 1
            push!(is, I)
            push!(js, indices[x, y + 1])
            push!(vs, 1.0 / h^2)
        end

        # south
        if y > 1
            push!(is, I)
            push!(js, indices[x, y - 1])
            push!(vs, 1.0 / h^2)
        end

        # east
        if x < n - 1
            push!(is, I)
            push!(js, indices[x + 1, y])
            push!(vs, 1.0 / h^2 + α / h)
        end

        # west
        if x > 1
            push!(is, I)
            push!(js, indices[x - 1, y])
            push!(vs, 1.0 / h^2 - α / h)
        end
    end

    return sparse(is, js, vs)
end

function setup(n, λ, α)
    # Δu + αuₓ + λeᵘ = 0 inside X
    # where u = 0        on ∂X.
    # Use X = [0, 1]² 
    # Discretize into n² tiles with (n+1)² grid points using central differences
    # so n - 1 central points.

    J = linear_part_sparse(n, α)

    return x -> J * x + λ .* exp.(x)
end

function mixer(f, x₀; σ = 0.3, max_iter = 10)
    F = zeros(length(x₀), max_iter)
    X = zeros(length(x₀), max_iter + 1)

    history = Float64[]

    # start out with fixed point iteration
    for k = 1:max_iter
        xₖ   = view(X, :, k    )
        xₖ₊₁ = view(X, :, k + 1)
        fₖ   = view(F, :, k)

        fₖ .= f(xₖ)

        ΔF = view(F, :, 2:k) .- view(F, :, 1:k-1)
        ΔX = view(X, :, 2:k) .- view(X, :, 1:k-1)

        qr_decomp = qr(ΔF)
        Q = Matrix(qr_decomp.Q)
        R = qr_decomp.R

        xₖ₊₁ .= xₖ .- σ * fₖ .+ σ .* (Q * (Q' * fₖ)) - ΔX * (R \ (Q' * fₖ))

        push!(history, norm(fₖ))
    end

    return X, history
end

function orthogonalize_and_normalize!(V::AbstractMatrix{T}, w::AbstractVector, h::AbstractVector) where {T}
    mul!(h, V', w)
    mul!(w, V, h, -1.0, 1.0)
    nrm = norm(w)

    # Constant used by ARPACK.
    η = one(real(T)) / √2

    projection_size = norm(h)

    # Repeat as long as the DGKS condition is satisfied
    # Typically this condition is true only once.
    while nrm < η * projection_size
        correction = V' * w
        projection_size = norm(correction)
        # w = w - V * correction
        mul!(w, V, correction, -1.0, 1.0)
        h .+= correction
        nrm = norm(w)
    end

    # Normalize; note that we already have norm(w).
    w .*= inv(nrm)

    nrm
end

function remove_first_col!(Q, R)
    n = size(R, 2)

    for i = 2:n
        c, s, nrm = givensAlgorithm(R[i - 1, i], R[i, i])
        R[i - 1, i] = nrm
        R[i    , i] = 0

        # Apply to R
        for j = i+1:n
            r₁ = R[i - 1, j]
            r₂ = R[i    , j]
    
            r₁′ =   c * r₁ + s * r₂
            r₂′ = -s' * r₁ + c * r₂
            
            R[i - 1, j] = r₁′
            R[i    , j] = r₂′
        end
        
        # Apply to Q
        for j = 1:size(Q, 1)
            q₁ = Q[j, i - 1]
            q₂ = Q[j, i    ]
    
            q₁′ = q₁ *  c + q₂ * s'
            q₂′ = q₁ * -s + q₂ * c
    
            Q[j, i - 1] = q₁′
            Q[j, i    ] = q₂′
        end
    end

    # Now shift R
    R[1:n-1, 1:n-1] .= R[1:n-1, 2:n]

    return nothing
end

using Test

function test_remove_first_col(n = 10)
    X = rand(100, n)
    decomp = qr(X)
    Q = Matrix(decomp.Q)
    R = decomp.R

    remove_first_col!(Q, R)

    @test norm(X[:, 2:end] .- Q[:, 1:n-1] * R[1:n-1,1:n-1]) < 1e-13
    @test norm(Q[:, 1:n-1]' * Q[:, 1:n-1] - I) < 1e-14
end

function improved_broyden_gram_schmidt(f, x₀; σ = 0.3, max_iter = 10, max_len = max_iter)
    ΔX = zeros(length(x₀), max_len)
    Q = zeros(length(x₀), max_len)
    R = zeros(max_len, max_len)

    fₖ, fₖ₋₁ = similar(x₀), similar(x₀)
    xₖ, xₖ₋₁ = similar(x₀), similar(x₀)

    xₖ .= x₀

    history = Float64[]

    # start out with fixed point iteration
    for k = 1:max_iter
        fₖ .= f(xₖ)

        # Remove the first column of the QR decomp
        if k - 1 > max_len
            remove_first_col!(Q, R)
            @views copyto!(ΔX[:, 1:max_len-1], ΔX[:, 2:max_len])
        end

        k′ = min(k - 1, max_len)

        # Gram-Schmidt Δfₖ₋₁ = fₖ - fₖ₋₁ w.r.t. previous ΔF
        # Repeated orthogonalization gives some stability and can be done with BLAS-2.
        if k > 1
            Q[:, k′] .= fₖ .- fₖ₋₁
            ΔX[:, k′] .= xₖ .- xₖ₋₁

            R[k′, k′] = @views orthogonalize_and_normalize!(Q[:, 1:k′-1], Q[:, k′], R[1:k′-1, k′])
        end

        @views Qₖ = Q[:, 1:k′]
        @views Rₖ = R[1:k′, 1:k′]
        @views ΔXₖ = ΔX[:, 1:k′]

        fₖ₋₁ .= fₖ
        xₖ₋₁ .= xₖ
        xₖ .= xₖ .- σ * fₖ .+ σ .* (Qₖ * (Qₖ' * fₖ)) - ΔXₖ * (Rₖ \ (Qₖ' * fₖ))

        push!(history, norm(fₖ))
    end

    return xₖ, history
end

function broyden(f, x₀, type=:good; σ = 0.3, max_iter = 10)
    n = length(x₀)
    J⁻¹ = Matrix(σ * I, n, n)

    fₖ, fₖ₋₁ = similar(x₀), similar(x₀)
    xₖ, xₖ₋₁ = similar(x₀), similar(x₀)

    xₖ .= x₀

    history = Float64[]

    for k = 1:max_iter
        fₖ .= f(xₖ)

        if k > 1
            Δx = xₖ - xₖ₋₁
            Δf = fₖ - fₖ₋₁

            if type === :good
                J⁻¹ = J⁻¹ + (Δx .- J⁻¹ * Δf) ./ dot(Δx, J⁻¹ * Δf) .* (Δx' * J⁻¹)
            else
                J⁻¹ = J⁻¹ + (Δx .- J⁻¹ * Δf) ./ dot(Δf, Δf) .* Δf'
            end
        end

        xₖ₋₁ .= xₖ
        fₖ₋₁ .= fₖ
        xₖ .-= J⁻¹ * fₖ

        push!(history, norm(fₖ))
    end

    return xₖ, history
end
