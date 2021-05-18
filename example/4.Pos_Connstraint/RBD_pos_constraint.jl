import Pkg;
Pkg.activate(joinpath(@__DIR__,"..","..")); Pkg.instantiate();
using TimerOutputs
using Test
include("../../src/RBD_floatingBase.jl")
# TODO: find out which function is required in MC version to run Altro
include("../../src/MC_floatingBase.jl")
"""Constants"""
##
Tf = 0.5
dt = 0.005
N = Int(Tf/dt)
ArmNumber = 3
link_pos_z_constraint = -2.4

"""Generate model"""
RBDmodel = FloatingSpaceOrthRBD(ArmNumber)

"""Run Altro"""
function solve_altro_test(RBDmodel, dt, N, link_pos_z_constraint)
    n,m = size(RBDmodel)
    n̄ = state_diff_size(RBDmodel)
    # trajectory 
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
    base_qf = RS.params(UnitQuaternion(RotZ(pi/2)))
    base_vf = [0., 0., 0.]
    base_ωf = [0., 0., 0.]
    joint_anglesf = fill.(pi/6,RBDmodel.nb)
    joint_wf = fill.(0.0,RBDmodel.nb)
    xf = [base_qf;base_xf;joint_anglesf;base_ωf;base_vf;joint_wf]

    #x0 and xf are the same as floatinbBaseSpace_trajOpt

    # objective
    Qf = Diagonal(@SVector fill(550., n))
    Q = Diagonal(@SVector fill(1e-1, n))
    R = Diagonal(@SVector fill(1e-4, m))
    costfuns = [TO.LieLQRCost(RobotDynamics.LieState(RBDmodel), Q, R, SVector{n}(xf); w=1) for i=1:N]
    costfuns[end] = TO.LieLQRCost(RobotDynamics.LieState(RBDmodel), Qf, R, SVector{n}(xf); w=550.0)
    obj = Objective(costfuns);

    to = TimerOutput()


    # constraints
    # Create Empty ConstraintList
    conSet = ConstraintList(n,m,N)
    max_pos = repeat([10.0,10.0,10.0],RBDmodel.nb)
    min_pos = repeat([-10.0,-10.0,link_pos_z_constraint],RBDmodel.nb)
    pos_limit = LinkPosConstraint(n,m, RBDmodel,max_pos,min_pos, TO.Inequality())
    # z = KnotPoint(x0,U0,0.005)
    # TO.evaluate(pos_limit, z)
    # ∇c = zeros(pos_limit.p, n+m)
    # TO.jacobian!(∇c, pos_limit, z)

    add_constraint!(conSet, pos_limit, 10:N-1)
    
    # problem
    prob = Problem(RBDmodel, obj, xf, tf, x0=x0, constraints=conSet);

    initial_controls!(prob, U_list)
    opts = SolverOptions(verbose=7, 
        static_bp=0, 
        square_root = true,
        iterations=1000, bp_reg=true,
        dJ_counter_limit = 1,
        iterations_inner = 30,
        cost_tolerance=1e-4, constraint_tolerance=1e-8,
        show_summary=true)
    altro = ALTROSolver(prob, opts)
    set_options!(altro, show_summary=true)
    solve!(altro);
    return altro
end
altro = solve_altro_test(RBDmodel, dt, N,link_pos_z_constraint)
# run it twice to get execution time
# altro = solve_altro_test(RBDmodel, dt, N,link_pos_z_constraint)

"""Visualization"""
n,m = size(RBDmodel)
X_list = states(altro)
U_list = controls(altro)
qs = []
vs = []
for idx = 1:N
    push!(qs,X_list[idx][1:(7 + ArmNumber)])
    push!(vs,X_list[idx][(8 + ArmNumber):end])
end
# view_sequence(RBDmodel, qs, vs)

"""Plot and save results"""

using Plots
using Plots.PlotMeasures
result_path = "results/4.pos_constraint/"
gr(size = (720, 450))
if (link_pos_z_constraint<-1)
    # no constraint
    file_name = "RBD_pos_no_constraint_"*string(ArmNumber)*"Arms.pdf"
    # plot z - pos of all arm links
    pos_list = zeros(N,RBDmodel.nb)
    for idx=1:N
        for link = 1:RBDmodel.nb
             pos = arm_world_pos(RBDmodel, [X_list[idx];zeros(m)], link) 
            pos_list[idx,link] = pos[3]
        end
    end

    label_list = ["link "*string(1)*" z pos"]
    for link=2:RBDmodel.nb
        label_list = hcat(label_list, "link "*string(link)*" z pos")
    end
    mytitle ="Minimal, link z positions with no constraint"
    plot(1:N, pos_list,title = mytitle, labels = label_list,fmt = :eps, legend=:topleft,
    size = (720, 350),
    bottom_margin = 3mm,
    top_margin = 3mm,
    left_margin = 3mm,
    xlabel = "Time steps", ylabel = "World frame position")

else
    # with constraint
    file_name = "RBD_pos_constraint_"*string(ArmNumber)*"Arms.pdf"
    # plot z - pos of all arm links
    pos_list = zeros(N,RBDmodel.nb+1)
    for idx=1:N
        for link = 1:RBDmodel.nb
            # pos_list[idx,3*(link-1).+(1:3)] .= arm_world_pos(RBDmodel, [X_list[idx];zeros(m)], link) 
            pos = arm_world_pos(RBDmodel, [X_list[idx];zeros(m)], link) 
            pos_list[idx,link] = pos[3]
        end
        pos_list[idx,RBDmodel.nb+1] = link_pos_z_constraint
    end

    label_list = ["link "*string(1)*" z pos"]
    for link=2:RBDmodel.nb
        label_list = hcat(label_list, "link "*string(link)*" z pos")
    end
    label_list = hcat(label_list, "z constraint ")
    
    mytitle ="Minimal, link z positions with constraint > "*string(link_pos_z_constraint)*"\n Final constraint violation 1.9774e-5"

    plot(1:N, pos_list,title = mytitle, labels = label_list,fmt = :eps, legend=:topleft,
         size = (720, 350),
         bottom_margin = 3mm,
         top_margin = 3mm,
         left_margin = 3mm,
         xlabel = "Time steps", ylabel = "World frame position")
end


savefig(result_path*file_name,)

# save altro stats
using JLD
save(result_path*file_name*".jld", 
    "X_list", X_list, 
    "U_list", U_list,
    "Total_iter", altro.stats.iterations,
    "Solve_time", altro.stats.tsolve,
    "Cost_hist", altro.stats.cost,
    "Solve_status", altro.stats.status)
Solve_status = load(result_path*file_name*".jld", "Solve_status")
