# in this script, we first simulate the floatingBaseSpace system, then we linearize the system dynamics 
# to get A B C G, then we calculate the K and d for this linearized problem. 
# basically this is a simplifed version of the inner loop of an iLQR algorithm

include("floatingBaseSpace.jl")
include("ecLQR/ecLQR.jl")

function simulate_model(model)
    n,m = size(model)
    np = model.p   # constraint size
    n̄ = RD.state_diff_size(model)

    x0 = generate_config(model, [0.0;0.0;0.0;0.0], fill.(0.0,model.nb));
    xf = generate_config(model, [1.3;0.3;1.0;pi/6], fill.(pi/6,model.nb));
    Tf = 3
    dt = 0.005
    N = Int(Tf/dt) 
    x_list = [SizedVector{n}(zeros(Float64,n)) for i=1:N]
    x_list[1] .= x0
    x = x0

    λ_list = [SizedVector{np}(zeros(Float64,np)) for i=1:N-1]
    λ_init = zeros(5*model.nb)
    λ = λ_init
    λ_list[1] .= λ_init
    
    u_list = [SizedVector{m}(zeros(Float64,m)) for i=1:N-1]
    U = 0.03*randn(m)
    println("start to simulate")
    @time begin
        for idx = 2:N
            println("step: ",idx)
            U .= 0.03*randn(m)
            u_list[idx-1] .= U
            x1, λ1 = discrete_dynamics(model,x, U, λ, dt)  # solved x1 is the state at t = idx
            x .= x1
            x_list[idx] .= x1
            λ .= λ1
            λ_list[idx-1] .= λ1   # solved λ1 is the λ at t= idx-1
        end 
    end
    # return x_list, u_list, λ_list


    # we have x_list, u_list, λ_list now 
    # next we need tools in TrajectoryOptimization to get some expansion data structure 
    # objective
    Qf = Diagonal(@SVector fill(550., n))
    Q = Diagonal(@SVector fill(1e-2, n))
    R = Diagonal(@SVector fill(1e-3, m))
    costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-1) for i=1:N]
    costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=550.0)
    obj = Objective(costfuns);

    # state difference
    Gjacob = [SizedMatrix{n,n̄}(zeros(n,n̄)) for k = 1:N+1]

    # expand cost function along the trajectory 
    # function TO.cost_expansion!(E::Objective, obj::Objective, Z::Traj, init::Bool=false, rezero::Bool=false)
    E = TO.QuadraticObjective(n̄,m,N)
	quad_exp = TO.QuadraticObjective(E, model)    # TO/src/objective.jl  create an objective with n,m

    # construct TO.Traj, follows TO/src/Problem.jl
    dt_list = fill(dt, N)
    t = pushfirst!(cumsum(dt_list), 0)
    Z = Traj(x_list,u_list,dt_list,t)

    TO.cost!(obj, Z)
    _J = TO.get_J(obj)
    println("Init cost is:", sum(_J))

    # following are from Altro/stc/ilqr/ilqr.jl
    # and 
    # Altro/src/ilqr/ilqr_solve.jl
    D = [TO.DynamicsExpansionMC(model) for k = 1:N-1]
    Λ = λ_list
    TO.dynamics_expansion!(RK4, D, model, Z, Λ)
    TO.state_diff_jacobian!(Gjacob, model, Z)
    TO.cost_expansion!(quad_exp, obj, Z, false, true)
    TO.error_expansion!(E, quad_exp, model, Z, Gjacob)

	# cost_exp = solver.E[k]
	# dyn_exp = solver.D[k]
    #    S⁺, s⁺ = S.Q, S.q
    # Q,q,R,r,H,c = cost_exp.Q,cost_exp.q,cost_exp.R,cost_exp.r,cost_exp.H,cost_exp.c
    # A,B,C,G = dyn_exp.A, dyn_exp.B, dyn_exp.C, dyn_exp.G

    α = 1.0/16
    # combine u and lambda together as input
    ec_zac = ecLQR{Float64}(n̄, m, np, np, N)
    # solve du and K d using ecLQR 
    @time ΔV_1, ΔV_2 = ecLQR_backward_Zac!(ec_zac, D, E)
    println("ec_zac | ΔV :", -α*ΔV_1)
    # # test ec_zac
    # constraint_vio = 0
    # for idx= 1:N-1
    #     dyn_exp = D[idx]
    #     A,B,C,G = dyn_exp.A, dyn_exp.B, dyn_exp.C, dyn_exp.G
    #     δx = 0.01*randn(n̄)
    #     δu = ec_zac.kl_list[idx] 
    #     mul!(δu, ec_zac.Kx_list[idx], δx, 1.0, 1.0)
    #     δλ = ec_zac.kλ_list[idx] 
    #     mul!(δλ, ec_zac.Kλ_list[idx], δx, 1.0, 1.0)
    #     constraint_vio += norm(G*(A*δx + B*δu + C*δλ))
    # end
    # println("ec_zac | total constraint vio:", constraint_vio)


    # update using ec_uλ
    ec = ec_zac

    # : rollout again, check dJ
    Z̄ = Traj(x_list,u_list,dt_list,t)
    Z̄[1].z = [x0; control(Z[1])]
    δx = zeros(n̄)
    δu = zeros(m)
    Λ2 = λ_list

    for k = 1:N-1
        δx .= RobotDynamics.state_diff(model, state(Z̄[k]), state(Z[k]))
        δu = α*ec.kl_list[k]
        mul!(δu, ec.Kx_list[k], δx, 1.0, 1.0)
        δλ = α*ec.kλ_list[k]
        mul!(δλ, ec.Kλ_list[k], δx, 1.0, 1.0)
        ū = control(Z[k]) + δu
        Λ2[k] = λ_list[k] + δλ
        # ū = control(Z[k])
        RobotDynamics.set_control!(Z̄[k], ū)
        x⁺, Λ2[k] = discrete_dynamics(model,state(Z̄[k]), control(Z̄[k]), Λ2[k], Z̄[k].dt)
        actual_δλ = Λ2[k] - λ_list[k]
        Z̄[k+1].z = [x⁺; control(Z[k+1])]

        # check δ terms

        # dyn_exp = D[k]
        # A,B,C,G = dyn_exp.A, dyn_exp.B, dyn_exp.C, dyn_exp.G
        # println(norm(G*(A*δx + B*δu + C*actual_δλ)))

    end

    TO.cost!(obj, Z̄)
    _J = TO.get_J(obj)
    println("after update using ec_zac, cost is:", sum(_J))

    # # compare the ΔV of the two cases
    return ec_zac
end




model = FloatingSpaceOrth(3)
ec_zac = simulate_model(model)
