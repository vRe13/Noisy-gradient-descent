using LinearAlgebra

# Normalize the first eigenvector
v1 = [0.5, 0.5]
v1 = v1 / norm(v1)  # Normalized to [1/√2, 1/√2]

# Create a second eigenvector that's orthogonal to v1
# The vector [-1, 0] is not orthogonal to v1, so we'll use Gram-Schmidt
v2_initial = [-1, 0]
v2 = v2_initial - dot(v2_initial, v1) * v1
v2 = v2 / norm(v2)  # Normalize

# Create matrix Q with orthonormal eigenvectors as columns
Q = [v1 v2]

# Create diagonal matrix with eigenvalues
Λ = Diagonal([1.0, 4.0])

# Construct the symmetric matrix
A = Q * Λ * Q'

# Check if the matrix is symmetric
println("Is symmetric: ", issymmetric(A))

# Display the matrix
println("Matrix A:")
display(A)

# Verify eigenvalues and eigenvectors
eigvals, eigvecs = eigen(A)
println("\nEigenvalues:")
display(eigvals)
println("\nEigenvectors:")
display(eigvecs)
println("Supposed unstable point;", sqrt(1 - 1/4)eigvecs[:, 1])
println("Supposed stable point;", sqrt(1 - 1/4)eigvecs[:, 2])