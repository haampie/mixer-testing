using SparseArrays, LinearAlgebra

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

function improved_broyden_gram_schmidt(f, x₀; σ = 0.3, max_iter = 10, max_len = max_iter)
    ΔX = zeros(length(x₀), max_iter)
    Q = zeros(length(x₀), max_iter)
    R = zeros(max_iter, max_iter)

    fₖ, fₖ₋₁ = similar(x₀), similar(x₀)
    xₖ, xₖ₋₁ = similar(x₀), similar(x₀)

    xₖ .= x₀

    history = Float64[]

    # start out with fixed point iteration
    for k = 1:max_iter
        fₖ .= f(xₖ)

        # Gram-Schmidt Δfₖ₋₁ = fₖ - fₖ₋₁ w.r.t. previous ΔF
        # Repeated orthogonalization gives some stability and can be done with BLAS-2.
        if k > 1
            Q[:, k - 1] .= fₖ .- fₖ₋₁
            ΔX[:, k - 1] .= xₖ .- xₖ₋₁

            R[k-1, k-1] = @views orthogonalize_and_normalize!(Q[:, 1:k-2], Q[:, k-1], R[1:k-2, k-1])

            @show R[k-1, k-1]
        end

        start = max(1, (k - 1) - max_len + 1)

        @views begin
            Qₖ = Q[:, start:k-1]
            Rₖ = R[start:k-1, start:k-1]
            ΔXₖ = ΔX[:, start:k-1]
        end

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
