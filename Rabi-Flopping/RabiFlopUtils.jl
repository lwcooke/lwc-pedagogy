"""
++ Rabi Flopping Utility Function Module ++

Contains functions for Monte-Carlo simulations of Atoms undergoing Rabi-Flopping including
various conditions
"""
module RabiFlopUtils

export H_rwa, solve_rabi_problem, quantum_state

using OrdinaryDiffEq, LinearAlgebra


const SIG_X = [0 1; 1 0]  # Pauli-x operator
const SIG_Y = [0 im; -im 0]  # Pauli-y operator
const SIG_Z = [1 0; 0 -1]  # Pauli-z operator


"""
    H_rwa(; Ω=1, ϕ=0, δ=0)

Hamiltonian for Rabi flopping in spin-1//2 system, after rotating-wave approximation

# Arguments:
- `Ω` : Rabi-frequency.
- `ϕ` : Phase of the driving field.
- `δ` : Detuning of the driving field from resonance
"""
function H_rwa(; Ω=1, ϕ=0, δ=0)
    return (Ω / 2) * (cos(ϕ) * SIG_X + sin(ϕ) * SIG_Y) + δ * SIG_Z
end


"""
    solve_rabi_problem(ψ0::Vector{ComplexF64}, tspan; H=H_rwa, kwargs...)

Solve the time-dependent Schrodinger equation from some initial state.

# Arguments:
- `ψ0::Vector{ComplexF64}` : Initial state. 
- `tspan` : Time interval to solve over, (initial, final).
- `H` : Hamiltonian to use, takes kwargs.
- `kwargs` : Parameters for the Hamiltonian.
"""
function solve_rabi_problem(ψ0::Vector{ComplexF64}, tspan; H=H_rwa, kwargs...)
    Schr = (ψ0, p, t) -> im * H(; p...) * ψ0
    prob = ODEProblem(Schr, ψ0, tspan, (; kwargs...))
    return solve(prob, Tsit5())
end


"""
    quantum_state(a, b, ϕ)

Construct normalized quantum state (Vector{ComplexF64}) from amplitudes a, b and phase ϕ.

"""
function quantum_state(a, b, ϕ)
    ψ = a * ComplexF64[1; 0] + b * exp(im * ϕ) * ComplexF64[0; 1]
    return ψ / norm(ψ)
end

end  # Module
