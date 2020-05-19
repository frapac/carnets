using DelimitedFiles
# Import data
data = readdlm("data_temp_GR4.txt")
# Time span ...
tspan = data[:, 1]
# ... and horizon
H = length(tspan)
# define some constant
begin const
    # Timestep = half an hour
    Δ = 1800.0
    c_air = 1.256e3
    # Max and min power
    Pᵤ = 1000.0
    Pₗ = 0.0
    # Max and min temperature
    Tₗ = 20.0
    Tᵤ = 22.0
    # Initial position
    T0 = 20.0
    # Up and low elec prices
    c_up = 0.18
    c_lo = 0.13
    # Take parameters as fitted by scipy
    p = [0.00255712 0.00075877]
    # Volume
    V = 1000.0
    # Coefficient of electrical heater
    ηₕ = 0.5
    p3 = ηₕ / V
end

# Build cost vector
cₜ = c_lo * ones(H)
# Full tariff between 7am and 11pm
cₜ[7 .<= tspan .% 24 .<= 23] .= c_up
# Rescale cost vector to get €/W
cₜ ./= 2 * 1000.0

# Copy for convenience
tₑₓₜ = data[:, 4]
Φₛ = data[:, 3]
tₑₓₜ .-= 10.0;

