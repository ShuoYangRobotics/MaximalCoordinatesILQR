import Pkg; Pkg.activate(joinpath(@__DIR__)); 
Pkg.instantiate();


using ConstrainedDynamics
using ConstrainedDynamicsVis
using ConstrainedControl
using StaticArrays
using LinearAlgebra
using Rotations

const RS = Rotations
const CD = ConstrainedDynamics
const CDV = ConstrainedDynamicsVis
const CC = ConstrainedControl

""" helper functions for simple two link system """
function fullargsinds(i)
    # x, v, q, ω
    return 13*(i-1) .+ (1:3), 
            13*(i-1) .+ (4:6), 
            13*(i-1) .+ (7:10), 
            13*(i-1) .+ (11:13)
end
function getStates(mech, sol=true)
    nb = length(mech.bodies)
    z = zeros(13*nb)
    for (i, body) in enumerate(mech.bodies)        
        xinds, vinds, qinds, ωinds = fullargsinds(i)
        f = sol ? CD.fullargssol : CD.posargsk
        z[xinds],z[vinds],q,z[ωinds] = f(body.state)
        z[qinds] = RS.params(q)
    end
    return z
end

# state config x v q w 
function generate_config(mech, body_pose, rotations)
    pin = zeros(3)
    pin[1] = body_pose[1]
    pin[2] = body_pose[2]
    pin[3] = body_pose[3]
    prev_q = UnitQuaternion(RotZ(body_pose[4]))
    state = [pin;zeros(3);RS.params(prev_q);zeros(3)]
    pin = pin+prev_q * [mech.bodies[1].shape.xyz[1]/2,0,0]
    for i = 1:length(rotations)
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [arm_width/2,0,0]
        link_x = pin+delta
        state = [state; link_x;zeros(3);RS.params(link_q);zeros(3)]

        prev_q = link_q
        pin += 2*delta
    end


    return state
end

# body pose is x y z thetaz, θ only have z rotation
function generate_config(mech, body_pose::Vector{<:Number}, θ::Vector{<:Number})
    rotations = []
    # joints are arranged orthogonally
    for i=1:length(θ)
        push!(rotations, UnitQuaternion(RotZ(θ[i])))
    end
    return generate_config(mech, body_pose, rotations)
end

# setposition setvelocity modifies state.xc qc..... 
function setStates!(mech, z)
    for (i, body) in enumerate(mech.bodies)   
        xinds, vinds, qinds, ωinds = fullargsinds(i)   
        setPosition!(body; x = SVector{3}(z[xinds]), q = UnitQuaternion(z[qinds]...))
        setVelocity!(body; v = SVector{3}(z[vinds]), ω = SVector{3}(z[ωinds]))
    end
end

function state_parts(mech, x, u)
    xd = SArray{Tuple{3},Float64,1,3}[]
    vd = SArray{Tuple{3},Float64,1,3}[]
    qd = UnitQuaternion{Float64}[]
    ωd = SArray{Tuple{3},Float64,1,3}[]
    Fτd = []
    for i=1:2
        xinds, vinds, qinds, ωinds = fullargsinds(i)
        push!(xd, x[xinds])
        push!(vd, x[vinds])
        push!(qd, UnitQuaternion(x[qinds]...))
        push!(ωd, x[ωinds])
    end

    push!(Fτd, SA[0.;0.;0.;0.;0.;0.])
    push!(Fτd, SA[u[1]])

    return xd, vd, qd, ωd, Fτd
end

""" Parameters """
timeStep = 10.0 # seconds

""" Define the base """
length1 = 0.5
width, depth = 0.5, 0.5
# Corner vectors
corners = [
    [[length1 / 2;length1 / 2;-length1 / 2]]
    [[length1 / 2;-length1 / 2;-length1 / 2]]
    [[-length1 / 2;length1 / 2;-length1 / 2]]
    [[-length1 / 2;-length1 / 2;-length1 / 2]]
    [[length1 / 2;length1 / 2;length1 / 2]]
    [[length1 / 2;-length1 / 2;length1 / 2]]
    [[-length1 / 2;length1 / 2;length1 / 2]]
    [[-length1 / 2;-length1 / 2;length1 / 2]]
]
# Initial orientation
##
ϕ1 = 0;
q1 = UnitQuaternion(RotZ(ϕ1))

# Define base link
origin = Origin{Float64}()
link0 = Box(width, depth, length1, 1., color = RGBA(1., 1., 0.))
link0.m = 10 # set base mass

# Constraints on base
# TODO: find what "Friction()" is doing
impacts = [InequalityConstraint(Friction(link0,[0;0;1.0], 0.01; p = corners[i])) for i=1:8] # above ground
world2base = EqualityConstraint(Floating(origin, link0)) # free floating

""" Define arms """
arm_length = 0.1
arm_width = 1.0
arm_depth = 0.1

# Define arm link
link1 = Box(arm_width, arm_depth, arm_length, arm_length, color = RGBA(0., 1., 0.))

# Constraints on the arms
joint1_axis = [0;0;1] # joint 1 rotates about z axis

vert01 = [width/2; 0; 0] # connection offset from link0 to joint1
vert11 = [-arm_width/2; 0; 0] # connection offset from link1 to joint1

joint1 = EqualityConstraint(Revolute(link0, link1, joint1_axis; p1=vert01,p2=vert11)) # joint1 : base to link1

# put them together
links = [link0; link1]
eqcs = [world2base; joint1]
ineqcs = impacts

mech = Mechanism(origin, links, eqcs, ineqcs) # TODO: this function is mutating!!!
# setPosition!(link0, x = [0.;0.;1.])
# setVelocity!(link0, v = [0;0;0], ω = [0;0;0])
# setPosition!(link1, x = [width/2;0.;1.])
# setVelocity!(link1, v = [0;0;0], ω = [0;0;0])


""" test state """
x0 = generate_config(mech, [0.0;0.0;1.0;pi/6], [pi/3]);
# x0 = generate_config(mech, [0.0;0.0;1.0;0.0], [0.0]);
u0 = [0]
reshape(x0,(13,2))'
# visualize state 
setStates!(mech,x0)
CD.discretizestate!(mech) # I finally found that this is very important
# each body.state has a lot of terms. setStates only modifies xc qc.. (continous state)
# this discretizestate! convert xc to xk (discretized states)
reshape(round.(getStates(mech)',digits=3),(13,2))'
steps = Base.OneTo(1)
storage = CD.Storage{Float64}(steps,length(mech.bodies))
for i=1:2
    storage.x[i][1] = mech.bodies[i].state.xc
    storage.v[i][1] = mech.bodies[i].state.vc
    storage.q[i][1] = mech.bodies[i].state.qc
    storage.ω[i][1] = mech.bodies[i].state.ωc
end
visualize(mech,storage, env = "editor")

bodyids = getid.(mech.bodies)
eqcids = getid.(mech.eqconstraints)
"""test constraint"""
# this is in CD->main_components->equalityConstraint.jl
constraint1 = CD.g(mech, geteqconstraint(mech, eqcids[2])) 
eqc = geteqconstraint(mech, eqcids[2])
c = CD.g(eqc.constraints[1], getbody(mech, eqc.parentid), getbody(mech, eqc.childids[1]), mech.Δt)# nonzero?
# typeof(eqc.constraints[1]) ===> ConstrainedDynamics.Translational{Float64,3}
# eqc.constraints[1] is a joint
# the above function should be CD->joints->abstract_joint.jl line 14
CD.constraintmat(eqc.constraints[2])
c2 = CD.g(eqc.constraints[2], getbody(mech, eqc.parentid).state, getbody(mech, eqc.childids[1]).state,  mech.Δt)# nonzero?
# the above function should be  CD->joints->joint.jl
# these function uses xk qk . Why state has some many different terms?
xa = CD.posargsnext(getbody(mech, eqc.parentid).state, mech.Δt)
xb = CD.posargsnext(getbody(mech, eqc.childids[1]).state, mech.Δt)
vertices = eqc.constraints[1].vertices # COM pos of front link and back link
CD.vrotate(xb[1] + CD.vrotate(vertices[2], xb[2]) - (xa[1] + CD.vrotate(vertices[1], xa[2])), inv(xa[2]))

""" test constraint jacobian """
xd, vd, qd, ωd, Fτd = state_parts(mech, x0,u0)
# function test(mechanism, eqc::EqualityConstraint{T,N,Nc}, Fτ::AbstractVector) where {T,N,Nc}
#     println(Nc)
#     println(N)
#     @assert length(Fτ)==3*Nc-N
# end

# for (i,id) in enumerate(eqcids)
#     test(mech, geteqconstraint(mech, id), Fτd[i])
# end
A1, B1, C1, G1 = CC.linearsystem(mech, xd, vd, qd, ωd, Fτd, bodyids, eqcids) 

dr = pi/19
x1 = generate_config(mech, [0.0;0.0;1.0;pi/6+dr], [pi/3+dr]);
xdp, vdp, qdp, ωdp, Fτd = state_parts(mech, x1,u0)
# visualize state 
setStates!(mech,x1)
CD.discretizestate!(mech) 

constraint2 = CD.g(mech, geteqconstraint(mech, eqcids[2])) 
# get the rotation error 
# x v q w 
state_error = zeros(24)
state_error[12*0 .+ (1:3)] = xdp[1]-xd[1]
state_error[12*0 .+ (4:6)] = vdp[1]-vd[1]
state_error[12*0 .+ (7:9)] = RS.rotation_error(qdp[1],qd[1], RS.QuatVecMap()) # Eqn 12  phi^{-1}(qdp*qd')
state_error[12*0 .+ (10:12)] = ωdp[1]-ωd[1]
state_error[12*1 .+ (1:3)] = xdp[2]-xd[2]
state_error[12*1 .+ (4:6)] = vdp[2]-vd[2]
state_error[12*1 .+ (7:9)] = RS.rotation_error(qdp[2],qd[2], RS.QuatVecMap())
state_error[12*1 .+ (10:12)] = ωdp[2]-ωd[2]

G1*state_error # should be very close to zero 

""" try my own constraint and jacobian """
function g(x,vertices)
    r_a = SVector{3}(x[13*0 .+ (1:3)])
    r_b = SVector{3}(x[13*1 .+ (1:3)])
    q_a = SVector{4}(x[13*0 .+ (7:10)])
    q_b = SVector{4}(x[13*1 .+ (7:10)])

    val = zeros(5)
    val[1:3] = (r_b + RS.vmat()*RS.rmult(q_b)'*RS.lmult(q_b)*RS.hmat()*vertices[2]) - 
    (r_a + RS.vmat()*RS.rmult(q_a)'*RS.lmult(q_a)*RS.hmat()*vertices[1])
    tmp = RS.vmat()*RS.lmult(q_a)'*q_b
    val[4:5] = tmp[1:2]
    return val
end
function Dg(x,vertices)
    # j87y69i
    q_a = SVector{4}(x[13*0 .+ (7:10)])
    q_b = SVector{4}(x[13*1 .+ (7:10)])   #  q_b[1] q_w,  q_b[2] q_v1,  q_b[3] q_v2,  q_b[4] q_v3
    Dgmtx = zeros(5,26)
    Dgmtx[:,13*0 .+ (1:3)] = [-I;zeros(2,3)]  # dg/dra
    Dgmtx[:,13*1 .+ (1:3)]  = [I;zeros(2,3)] # dg/drb
    Dgmtx[:,13*0 .+ (7:10)] = [-2*RS.vmat()*RS.rmult(q_a)'*RS.rmult(RS.hmat()*vertices[1]);
                            [q_b[3]  q_b[4] -q_b[1] -q_b[2];
                             q_b[2] -q_b[1] -q_b[4] q_b[3]]
                           ]
    Dgmtx[:,13*1 .+ (7:10)] = [2*RS.vmat()*RS.rmult(q_b)'*RS.rmult(RS.hmat()*vertices[2]);
                            [-q_a[3] -q_a[4]  q_a[1] q_a[2];
                             -q_a[2]  q_a[1]  q_a[4] -q_a[3]]
                           ]
    return Dgmtx
end
gval = g(x0,vertices)
Dgmtx = Dg(x0,vertices)

# this is called state_diff_jacobian in Altro
q_a0 = SVector{4}(x0[13*0 .+ (7:10)])
q_b0 = SVector{4}(x0[13*1 .+ (7:10)])
sdJ0 = zeros(26,24)
sdJ0[13*0 .+ (1:3), 12*0 .+ (1:3)] = I(3)
sdJ0[13*0 .+ (4:6), 12*0 .+ (4:6)] = I(3)
sdJ0[13*0 .+ (7:10), 12*0 .+ (7:9)] = RS.∇differential(UnitQuaternion(q_a0))
sdJ0[13*0 .+ (11:13), 12*0 .+ (10:12)] = I(3)
sdJ0[13*1 .+ (1:3), 12*1 .+ (1:3)] = I(3)
sdJ0[13*1 .+ (4:6), 12*1 .+ (4:6)] = I(3)
sdJ0[13*1 .+ (7:10), 12*1 .+ (7:9)] = RS.∇differential(UnitQuaternion(q_b0))
sdJ0[13*1 .+ (11:13), 12*1 .+ (10:12)] = I(3)

q_a1 = SVector{4}(x1[13*0 .+ (7:10)])
q_b1 = SVector{4}(x1[13*1 .+ (7:10)])
sdJ1 = zeros(26,24)
sdJ1[13*0 .+ (1:3), 12*0 .+ (1:3)] = I(3)
sdJ1[13*0 .+ (4:6), 12*0 .+ (4:6)] = I(3)
sdJ1[13*0 .+ (7:10), 12*0 .+ (7:9)] = RS.∇differential(UnitQuaternion(q_a1))
sdJ1[13*0 .+ (11:13), 12*0 .+ (10:12)] = I(3)
sdJ1[13*1 .+ (1:3), 12*1 .+ (1:3)] = I(3)
sdJ1[13*1 .+ (4:6), 12*1 .+ (4:6)] = I(3)
sdJ1[13*1 .+ (7:10), 12*1 .+ (7:9)] = RS.∇differential(UnitQuaternion(q_b1))
sdJ1[13*1 .+ (11:13), 12*1 .+ (10:12)] = I(3)

Dgmtx*sdJ1*state_error

"""test A B C jaocbians"""


""" Define flotation force through controller """
baseid = eqcs[1].id # this is the floating base, as a free joint
mech.g = 0 # disable gravity
function controller!(mechanism, k)
    # F = SA[0;0; 9.81 * 10.4]
    F = SA[0;0; 0]
    τ = SA[0;0;0.0]
    setForce!(mechanism, geteqconstraint(mechanism,baseid), [F;τ])
    return
end
""" Start simulation """
# storage = simulate!(mech, timeStep, record = true)
# visualize(mech, storage, env = "editor")


