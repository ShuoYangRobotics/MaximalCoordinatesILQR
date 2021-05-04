import Pkg; Pkg.activate(joinpath(@__DIR__)); 
Pkg.instantiate();
include("RBDmodel.jl")

RBDmodel = FloatingSpaceOrthRBD(3)

"""Simulate one step"""
Tf = 0.01
dt = 0.01
base_x0 = [0., 0., 1.]
base_q0 = RS.params(UnitQuaternion(RotZ(pi/3)))
base_v0 = [0., 0., 0.]
base_ω0 = [0., 0., 0.]
joint_angles0 = fill.(pi/13,RBDmodel.nb)
U = 0.03*zeros(6+RBDmodel.nb)
U[5] = 1
U[6] = 1

rcstate = RBD.MechanismState(RBDmodel.tree)
RBD.set_configuration!(rcstate, joints(RBDmodel.tree)[1], base_q0) # set floating joint rotation
RBD.set_velocity!(rcstate, joints(RBDmodel.tree)[1], [base_ω0; base_v0])

for idx = 1 : RBDmodel.nb  
    RBD.set_configuration!(rcstate, joints(RBDmodel.tree)[idx+1], joint_angles0[idx])
    RBD.set_velocity!(rcstate, joints(RBDmodel.tree)[idx+1], 0)
end

RBD.set_configuration!(rcstate, RBD.configuration(rcstate) .+ [zeros(4);base_x0;zeros(RBDmodel.nb)])

controller! = function controller!(τ, t, rcstate)
    τ .= 0.03*zeros(6 + RBDmodel.nb)
    τ[2] = U[5]
    τ[3] = U[6]
end
rcstate.q
@show oldstate = copy(rcstate.q)
ts, qs, vs = RBD.simulate(rcstate, Tf, controller!, Δt=dt)
# @test rcstate.q == oldstate this shows the state is mutating

"""Using discrete_dynamics"""
z = KnotPoint([qs[1];vs[1]...],[U[4:6];U[1:3];U[7:end]],dt)
xnext = discrete_dynamics(RBDmodel, z)

using Test
@test xnext[1:10] ≈ qs[2]
@test xnext[11:19] ≈ vs[2]


""" Test Jacobian """
∇f = RobotDynamics.DynamicsJacobian(RBDmodel)
RobotDynamics.discrete_jacobian!(RK3, ∇f, RBDmodel, z)
xlinear = (∇f.A)*state(z)*z.dt+ ∇f.B*control(z)*z.dt
@show xlinear - xnext

""" Test attitude Jacbian """
n,m = size(RBDmodel)
n̄ = state_diff_size(RBDmodel)
G = zeros(n,n̄ )
RobotDynamics.state_diff_jacobian!(G, RBDmodel, z)
G


""" Test Altro """

using Altro
using TrajectoryOptimization
n,m = size(RBDmodel)
n̄ = state_diff_size(RBDmodel)
# trajectory 
N = 100   
dt = 0.005                  # number of knot points
tf = (N-1)*dt           # final time

U0 = @SVector fill(0.00001, m)
U_list = [U0 for k = 1:N-1]
