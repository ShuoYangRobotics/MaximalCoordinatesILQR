import Pkg;
Pkg.activate(joinpath(@__DIR__,"..","..")); Pkg.instantiate();
using TimerOutputs
using Test
include("../../src/RBD_floatingBase.jl")

"""Constants"""
Tf = 5
dt = 0.005
N = Int(Tf/dt)
ArmNumber = 3

"""Generate model"""
RBDmodel = FloatingSpaceOrthRBD(ArmNumber)

"""Initial conditions"""
base_x0 = [0., 0., 1.]
base_q0 = RS.params(UnitQuaternion(RotZ(pi/3)))
base_v0 = [0., 0., 0.]
base_ω0 = [0., 0., 0.]
joint_angles0 = fill.(0.,RBDmodel.nb)

"""Input sequence"""
U = 0.03*zeros(6 + RBDmodel.nb)
# U = [Base_forces; Base_torques; Joint_torque]
# U is defined in body frame
U[1] = 1
U[4] = 1

"""Set initial conditions"""
rcstate = RBD.MechanismState(RBDmodel.tree)
RBD.set_configuration!(rcstate, joints(RBDmodel.tree)[1], base_q0) # set floating joint rotation
RBD.set_velocity!(rcstate, joints(RBDmodel.tree)[1], [base_ω0; base_v0])
for idx = 1 : RBDmodel.nb  
    RBD.set_configuration!(rcstate, joints(RBDmodel.tree)[idx+1], joint_angles0[idx])
    RBD.set_velocity!(rcstate, joints(RBDmodel.tree)[idx+1], 0)
end
RBD.set_configuration!(rcstate, RBD.configuration(rcstate) .+ [zeros(4);base_x0;zeros(RBDmodel.nb)])

view_single_state(RBDmodel, configuration(rcstate), velocity(rcstate))

"""Define controller"""
controller! = function controller!(τ, t, rcstate)
    τ .= 0.03*zeros(6 + RBDmodel.nb)
    # U = [Base_forces; Base_torques; Joint_torque]
    # τ = [Base_torques; Base_forces; Joint_torque]
    τ[1:3] .= U[4:6]
    τ[4:6] .= U[1:3]
end

"""Run simulation"""
ts, qs, vs = RBD.simulate(rcstate, Tf, controller!, Δt=dt)
view_sequence(RBDmodel, qs, vs)

"""Using discrete_dynamics"""
# state (q) q x θ  (v) w  v  ̇θ̇    control   τ   F    jointτ
# initialize a knotpoint using the first state and controls
z = KnotPoint([qs[1];vs[1]...],[U[4:6];U[1:3];U[7:end]],dt)
xnext = RD.discrete_dynamics(RBDmodel, z)
# they should be the same
@test xnext[1:10] ≈ qs[2]
@test xnext[11:19] ≈ vs[2]

""" Test Jacobian """
∇f = RobotDynamics.DynamicsJacobian(RBDmodel)
RobotDynamics.discrete_jacobian!(RK4, ∇f, RBDmodel, z)
xlinear = (∇f.A)*state(z)*z.dt+ ∇f.B*control(z)*z.dt
@show xlinear - xnext

""" Test attitude Jacbian """
n,m = size(RBDmodel)
n̄ = state_diff_size(RBDmodel)
G = zeros(n,n̄ )
RobotDynamics.state_diff_jacobian!(G, RBDmodel, z)
G
