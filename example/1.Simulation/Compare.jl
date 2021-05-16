import Pkg;
Pkg.activate(joinpath(@__DIR__,"..","..")); Pkg.instantiate();
using TimerOutputs
using Test
include("../../src/MC_floatingBase.jl")
include("../../src/RBD_floatingBase.jl")

"""Constants"""
Tf = 5
dt = 0.005
N = Int(Tf/dt)
ArmNumber = 3

"""Generate model"""
MCmodel = FloatingSpaceOrth(ArmNumber)
RBDmodel = FloatingSpaceOrthRBD(ArmNumber)

"""Initial conditions"""
base_x0 = [0., 0., 1.]
base_q0 = RS.params(UnitQuaternion(RotZ(pi/2)))
base_v0 = [0., 0., 0.]
base_ω0 = [0., 0., 0.]
joint_angles0 = fill.(pi/13,RBDmodel.nb)

"""Input sequence"""
U = 0.03*zeros(6 + MCmodel.nb)
# U = [Base_forces; Base_torques; Joint_torque]
U[1] = 1

"""Set initial conditions"""
# MC
x0 = generate_config(MCmodel, [base_x0;pi/3], joint_angles0);
# RBD
rcstate = RBD.MechanismState(RBDmodel.tree)
RBD.set_configuration!(rcstate, joints(RBDmodel.tree)[1], base_q0) # set floating joint rotation
RBD.set_velocity!(rcstate, joints(RBDmodel.tree)[1], [base_ω0; base_v0])
for idx = 1 : RBDmodel.nb  
    RBD.set_configuration!(rcstate, joints(RBDmodel.tree)[idx+1], joint_angles0[idx])
    RBD.set_velocity!(rcstate, joints(RBDmodel.tree)[idx+1], 0)
end
RBD.set_configuration!(rcstate, RBD.configuration(rcstate) .+ [zeros(4);base_x0;zeros(RBDmodel.nb)])

"""Define controller"""
RBDcontroller! = function RBDcontroller!(τ, t, rcstate)
    τ .= 0.03*zeros(6 + RBDmodel.nb)
    # U = [Base_forces; Base_torques; Joint_torque]
    # τ = [Base_torques; Base_forces; Joint_torque]
    τ[1:3] .= U[4:6]
    τ[4:6] .= U[1:3]
    τ[7:end] .= U[7:end]
end

MCcontroller! = function MCcontroller!(τ, t, mcstate)
    τ .= U
end

"""Run simulation"""
# RC
ts, qs, vs = RBD.simulate(rcstate, Tf, RBDcontroller!, Δt=dt)
view_sequence(RBDmodel, qs, vs)
# MC
states = simulate(MCmodel, x0, Tf, MCcontroller!, dt)
view_sequence(MCmodel, states)


# compare base position
@test qs[1][5:7] ≈ states[1][1:3]
@test qs[2][5:7] ≈ states[2][1:3]
@test qs[N][5:7] ≈ states[N][1:3]
