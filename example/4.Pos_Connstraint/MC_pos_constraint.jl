import Pkg;
Pkg.activate(joinpath(@__DIR__,"..","..")); Pkg.instantiate();
using TimerOutputs
using Test
include("../../src/MC_floatingBase.jl")

"""Constants"""
Tf = 0.5
dt = 0.005
N = Int(Tf/dt)
ArmNumber = 3
link_pos_z_constraint = -2.04

"""Generate model"""
MCmodel = FloatingSpaceOrth(ArmNumber)

# run test to trigger model function compile
test_dyn()

# put solve steps in function 
function solve_altro_test(model, N, dt,link_pos_z_constraint)
    # trajectory 
    tf = (N-1)*dt           # final time
    n,m = size(model)

    U0 = @SVector fill(0.00001, m)
    U_list = [U0 for k = 1:N-1]

    x0 = generate_config(model, [0.01;0.01;0.01;0.01], fill.(0.01,model.nb))

    xf = generate_config(model, [0.3;0.3;1.0;pi/4], fill.(pi/6,model.nb))

    # objective
    Qf = Diagonal(@SVector fill(450., n))
    Q = Diagonal(@SVector fill(1e-2, n))
    R = Diagonal(@SVector fill(1e-4, m))
    costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1) for i=1:N]
    costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=550.0)
    obj = Objective(costfuns);

    # constraints
    # Create Empty ConstraintList
    conSet = ConstraintList(n,m,N)
    
    # limit the pos of every arm link 
    max_pos = repeat([10.0,10.0,10.0],model.nb)
    min_pos = repeat([-10.0,-10.0,link_pos_z_constraint],model.nb)

    p = 3*model.nb
    A = zeros(p,n+m)
    b = zeros(p)
    b2 = zeros(p)
    for idx = 2:model.nb+1
        statea_inds!(model, idx)
        A[3*(idx-2).+(1:3),model.r_ainds] = I(3)
        b[3*(idx-2).+(1:3)] .= max_pos[3*(idx-2).+(1:3)]
        b2[3*(idx-2).+(1:3)] .= min_pos[3*(idx-2).+(1:3)]
    end

    lin_upper = LinearConstraint(n,m,A,b, TO.Inequality())
    lin_lower = LinearConstraint(n,m,-A,-b2, TO.Inequality())

    add_constraint!(conSet, lin_lower, 1:N-1)
    add_constraint!(conSet, lin_upper, 1:N-1)

    to = TimerOutput()
    # problem
    prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet);

    initial_controls!(prob, U_list)
    opts = SolverOptions(verbose=7, 
        static_bp=0, 
        square_root = true,
        iterations=80, bp_reg=true,
        dJ_counter_limit = 1,
        iterations_inner = 30,
        cost_tolerance=1e-4, constraint_tolerance=1e-9)
    altro = ALTROSolver(prob, opts)
    set_options!(altro, show_summary=true)
    solve!(altro);
    return altro
end
altro = solve_altro_test(MCmodel, N, dt,link_pos_z_constraint)
# run it twice to get execution time
# altro = solve_altro_test(MCmodel, N, dt,link_pos_z_constraint)

"""Visualization"""
n,m = size(MCmodel)
X_list = states(altro)
U_list = controls(altro)

# a final simulation pass to get "real" state trajectory
# λ_init = zeros(5*MCmodel.nb)
# λ = λ_init
# Xfinal_list = copy(X_list)
# Xfinal_list[1] = SVector{n}(X_list[1])
# mech = vis_mech_generation(MCmodel)
# for idx = 1:N-1
#     x1, λ1 = discrete_dynamics(MCmodel,Xfinal_list[idx], U_list[idx], λ, dt)
#     setStates!(MCmodel,mech,x1)
#     Xfinal_list[idx+1] = SVector{n}(x1)
#     λ = λ1
# end
# view_sequence(MCmodel, Xfinal_list)

"""Plot and save results"""

using Plots
using Plots.PlotMeasures
result_path = "results/4.pos_constraint/"

if (link_pos_z_constraint<-1)
    # no constraint
    file_name = "MC_pos_no_constraint_"*string(ArmNumber)*"Arms.pdf"

    # plot z - pos of all arm links
    pos_list = zeros(N,MCmodel.nb)
    for idx=1:N
        for link = 1:MCmodel.nb
            statea_inds!(MCmodel, link+1)
            pos_list[idx,link] = X_list[idx][MCmodel.r_ainds[3]]
        end
    end

    label_list = [string(1)*"z"]
    for link=2:MCmodel.nb
        label_list = hcat(label_list, string(link)*"z")
    end
    mytitle ="Maximal, link z positions with no constraint"
    plot(1:N, pos_list,title = mytitle, labels = label_list,fmt = :eps, legend=:topleft,
    size = (720, 350),
    bottom_margin = 3mm,
    top_margin = 3mm,
    left_margin = 3mm,
    xlabel = "Time steps", ylabel = "World frame position")
else
    # with constraint
    file_name = "MC_pos_constraint_"*string(ArmNumber)*"Arms.pdf"

    # plot z - pos of all arm links
    pos_list = zeros(N,MCmodel.nb+1)
    for idx=1:N
        for link = 1:MCmodel.nb
            statea_inds!(MCmodel, link+1)
            pos_list[idx,link] = X_list[idx][MCmodel.r_ainds[3]]
        end
        pos_list[idx,MCmodel.nb+1] = link_pos_z_constraint
    end

    label_list = [string(1)*"z"]
    for link=2:MCmodel.nb
        label_list = hcat(label_list, string(link)*"z")
    end
    label_list = hcat(label_list, "z constraint ")
    mytitle ="Maximal, link z positions with constraint > "*string(link_pos_z_constraint)*"\n Final constraint violation 0.0"

    plot(1:N, pos_list,title = mytitle, labels = label_list,fmt = :eps, legend=:topleft,
         size = (720, 350),
         bottom_margin = 3mm,
         top_margin = 3mm,
         left_margin = 3mm,
         xlabel = "Time steps", ylabel = "World frame position")
end
savefig(result_path*file_name)




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
