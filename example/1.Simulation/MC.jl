import Pkg;
Pkg.activate(joinpath(@__DIR__,"..","..")); Pkg.instantiate();
using TimerOutputs
using Test
include("../../src/MC_floatingBase.jl")

"""Constants"""
Tf = 5
dt = 0.005
N = Int(Tf/dt)
ArmNumber = 3

"""Generate model"""
MCmodel = FloatingSpaceOrth(ArmNumber)

"""Initial conditions"""
base_x0 = [0., 0., 1.]
base_q0 = RS.params(UnitQuaternion(RotZ(pi/3)))
base_v0 = [0., 0., 0.]
base_ω0 = [0., 0., 0.]
joint_angles0 = fill.(0.,MCmodel.nb)

"""Input sequence"""
U = 0.03*zeros(6 + MCmodel.nb)
# U = [Base_forces; Base_torques; Joint_torque]
# U is defined in body frame
U[1] = 1
U[4] = 1

"""Set initial conditions"""
x0 = generate_config(MCmodel, [base_x0;pi/3], joint_angles0);
view_single_state(MCmodel, x0)

"""Define controller"""
controller! = function controller!(τ, t, mcstate)
    τ[1:3] .= UnitQuaternion(mcstate[7:10]...) * U[1:3] # rotate from body frame to world frame
    τ[4:6] .= U[4:6]
    τ[7:end] .= U[7:end]
end

"""Run simulation"""
states = simulate(MCmodel, x0, Tf, controller!, dt)
view_sequence(MCmodel, states)