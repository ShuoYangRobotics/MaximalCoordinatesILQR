import Pkg; Pkg.activate(joinpath(@__DIR__)); 
Pkg.instantiate();
using TimerOutputs
using Test
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

"""Using discrete_dynamics"""
# state (q) q x θ  (v) w  v  ̇θ̇    control   τ   F    jointτ
# initialize a knotpoint using the first state
z = KnotPoint([qs[1];vs[1]...],[U[4:6];U[1:3];U[7:end]],dt)
xnext = discrete_dynamics(RBDmodel, z)
# they should be the same
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
n,m = size(RBDmodel)
n̄ = state_diff_size(RBDmodel)
# trajectory 
N = 100   
dt = 0.005                  # number of knot points
tf = (N-1)*dt           # final time

U0 = @SVector fill(0.00001, m)
U_list = [U0 for k = 1:N-1]

base_x0 = [0.01, 0.01, 0.01]
base_q0 = RS.params(UnitQuaternion(RotZ(0.01)))
base_v0 = [0., 0., 0.]
base_ω0 = [0., 0., 0.]
joint_angles0 = fill.(0.01,RBDmodel.nb)
joint_w0 = fill.(0.0,RBDmodel.nb)
x0 = [base_q0;base_x0;joint_angles0;base_ω0;base_v0;joint_w0]

base_xf = [0.3;0.3;1.0]
base_qf = RS.params(UnitQuaternion(RotZ(pi/6)))
base_vf = [0., 0., 0.]
base_ωf = [0., 0., 0.]
joint_anglesf = fill.(pi/6,RBDmodel.nb)
joint_wf = fill.(0.0,RBDmodel.nb)
xf = [base_qf;base_xf;joint_anglesf;base_ωf;base_vf;joint_wf]

#x0 and xf are the same as floatinbBaseSpace_trajOpt

# objective
Qf = Diagonal(@SVector fill(550., n))
Q = Diagonal(@SVector fill(1e-2, n))
R = Diagonal(@SVector fill(1e-3, m))
costfuns = [TO.LieLQRCost(RobotDynamics.LieState(RBDmodel), Q, R, SVector{n}(xf); w=1e-1) for i=1:N]
costfuns[end] = TO.LieLQRCost(RobotDynamics.LieState(RBDmodel), Qf, R, SVector{n}(xf); w=550.0)
obj = Objective(costfuns);

# constraints
# Create Empty ConstraintList
conSet = ConstraintList(n,m,N)

const to = TimerOutput()
# # problem
prob = Problem(RBDmodel, obj, xf, tf, x0=x0, constraints=conSet);

initial_controls!(prob, U_list)
opts = SolverOptions(verbose=7, 
    static_bp=0, 
    square_root = true,
    iterations=150, bp_reg=true,
    cost_tolerance=1e-4, constraint_tolerance=1e-4)
altro = ALTROSolver(prob, opts)
set_options!(altro, show_summary=true)
solve!(altro);
aaa = 1;

# to visualize
include("floatingBaseSpace.jl")
X_list = states(altro)
mc_model = FloatingSpaceOrth(3)
mc_n,mc_m = size(mc_model)
mech = vis_mech_generation(mc_model)
steps = Base.OneTo(Int(N))
storage = CD.Storage{Float64}(steps,length(mech.bodies))
for idx=1:N
    mc_state = generate_config_rc2mc(mc_model, X_list[idx][5:7], X_list[idx][1:4], zeros(3), zeros(3), X_list[idx][8:7+mc_model.nb])
    setStates!(mc_model,mech,mc_state)
    for i=1:mc_model.nb+1
        storage.x[i][idx] = mech.bodies[i].state.xc
        storage.v[i][idx] = mech.bodies[i].state.vc
        storage.q[i][idx] = mech.bodies[i].state.qc
        storage.ω[i][idx] = mech.bodies[i].state.ωc
    end
end
visualize(mech,storage, env = "editor")


"""Constants"""
struct EFVConstraint{S,T} <: TO.StageConstraint
	n::Int
	m::Int
	model::FloatingSpaceRBD{T}
    maxV::Float64
	sense::S
	function EFVConstraint(n::Int, m::Int, model::FloatingSpaceRBD{T}, maxV::Float64,
			sense::TO.ConstraintSense) where {T}
		new{typeof(sense), T}(n,m,model,maxV,sense)
	end
end
TO.sense(con::EFVConstraint) = con.sense
TO.state_dim(con::EFVConstraint) = con.n
TO.control_dim(con::EFVConstraint) = con.m
function TO.evaluate(con::EFVConstraint, z::AbstractKnotPoint)
    iq, ix, iθ, iω, iv, iϕ = state_parts(RBDmodel)
    thetalist = state(z)[iθ]
    ϕlist = state(z)[iϕ]
    return MRB.FKinSpace(con.model.M, con.model.Slist, thetalist)[1:3,1:3]*(MRB.JacobianBody(con.model.Blist, thetalist)*ϕlist)[4:6] + state(z)[iv] - ones(3) * con.maxV
end

@testset "Constraints" begin
    lin_upper = EFVConstraint(n,m,RBDmodel,3.0, TO.Inequality())

    base_x = [0.01, 0.01, 0.01]
    base_q = RS.params(UnitQuaternion(RotZ(0.01)))
    base_v = [5., 5., 5.]
    base_ω = [0., 0., 0.]
    joint_angles = fill.(0.01,RBDmodel.nb)
    joint_w = fill.(0.0,RBDmodel.nb)
    x1 = [base_q;base_x;joint_angles;base_ω;base_v;joint_w]
    z1 = KnotPoint(x1,zeros(6+RBDmodel.nb),dt)
    @show TO.evaluate(lin_upper, z1)
    @test min(TO.evaluate(lin_upper, z1)...) > 0

    base_x = [0.01, 0.01, 0.01]
    base_q = RS.params(UnitQuaternion(RotZ(0.01)))
    base_v = [0., 0., 0.]
    base_ω = [0., 0., 0.]
    joint_angles = fill.(0.,RBDmodel.nb)
    joint_w = [0., 0., 10.]
    x2 = [base_q;base_x;joint_angles;base_ω;base_v;joint_w]
    z2 = KnotPoint(x2,zeros(6+RBDmodel.nb),dt)
    @show TO.evaluate(lin_upper, z2)
    @test max(TO.evaluate(lin_upper, z2)...) > 0
end
# function TO.jacobian!(∇c, con::EFVConstraint, z::AbstractKnotPoint)
# 	∇c[:,con.inds[1]:con.inds[end]] .= con.A
# 	return true
# end

# function TO.change_dimension(con::EFVConstraint, n::Int, m::Int, ix=1:n, iu=1:m)
# 	inds0 = [ix; n .+ iu]  # indices of original z in new z
# 	inds = inds0[con.inds] # indices of elements in new z
# 	EFVConstraint(n, m, con.A, con.b, con.sense, inds)
# end
lin_upper = EFVConstraint(n,m,RBDmodel,3.0, TO.Inequality())
function evlatz(z::AbstractVector{T}) where T
    thetalist = z[iθ]
    ϕlist = z[iϕ]
    Vector(SVector(MRB.FKinSpace(RBDmodel.M, RBDmodel.Slist, thetalist)[1:3,1:3]*(MRB.JacobianBody(RBDmodel.Blist, thetalist)*ϕlist)[4:6]))
end
z = x0
ForwardDiff.jacobian(evlatz, z)