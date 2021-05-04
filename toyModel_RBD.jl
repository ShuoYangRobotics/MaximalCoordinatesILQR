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

ts, qs, vs = RBD.simulate(rcstate, Tf, controller!, Δt=dt)

qs[1]
vs[1]
qs[2]
vs[2]
"""Using discrete_dynamics"""
z = KnotPoint([qs[1];vs[1]...],U,dt)

ret = discrete_dynamics(RBDmodel, z)
@show ret.q̇
@show qs[1] + ret.q̇ * dt
@show qs[2]
ret.v̇