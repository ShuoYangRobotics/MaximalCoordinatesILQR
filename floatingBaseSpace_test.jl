
    # @testset "basic construct test" begin
    #     # basic construct test
    #     a = FloatingSpace()
    #     @test Altro.config_size(a) == 14
    #     println(Lie_P(a))
    # end

    # # test rotations
    # @testset "rotation function" begin
    #     model = FloatingSpace()
    #     q = UnitQuaternion(SVector{4}([1;2;3;4.0]),false)
    #     q_rmult!(model,RS.params(q))
    #     qq = RS.rmult(q)
    #     @test qq ≈ model.rmat
    #     q_lmult!(model,RS.params(q))
    #     qq = RS.lmult(q)
    #     @test qq ≈ model.lmat

    #     q_rmult_T!(model,RS.params(q))
    #     qq = RS.rmult(q)'
    #     @test qq ≈ model.rmat
    #     q_lmult_T!(model,RS.params(q))
    #     qq = RS.lmult(q)'
    #     @test qq ≈ model.lmat
    # end

    # # test: constraint g
    # @testset "constraint g" begin
    #     model = FloatingSpace(3)
    #     x0 = generate_config_with_rand_vel(model, [2.0;2.0;1.0;pi/4], fill.(pi/4,model.nb))
    #     gval = g(model,x0)
    #     # println(gval)
    #     @time g!(model,x0)
    #     # println(model.g_val)
    #     @test Array(gval) ≈ Array(model.g_val) atol=1e-9

    #     Dgmtx = Dg(model,x0)
    #     @time Dg!(model,x0)
    #     # println(model.Dgmtx)
    #     # println(Dgmtx)
    #     @test Array(Dgmtx) ≈ Array(model.Dgmtx) atol=1e-9

    #     # # TODO, test gp1 and Dgp1
    #     gval = gp1(model,x0,0.01)
    #     # println(gval)
    #     @time gp1!(model,x0,0.01)
    #     @test Array(gval) ≈ Array(model.g_val)  atol=1e-9
        
    #     Dp1gmtx = Dgp1(model,x0,0.01)
    #     @time Dgp1!(model,x0,0.01)
    #     @test Array(Dp1gmtx) ≈ Array(model.Dgmtx)  atol=1e-9
    #     # # println(Dgmtx)
    #     gp1aug(z) = gp1(model,z,0.01)
    #     Dgp1forward = ForwardDiff.jacobian(gp1aug,x0)
    #     @test Array(Dgp1forward) ≈ Array(Dp1gmtx) atol=1e-3

    #     q_a = UnitQuaternion(RotX(0.03))
    #     q_b = UnitQuaternion(RotY(0.03))
    #     vertices = [1,2,3,4,5,6]
    #     joint_direction = [0,0,1]
    #     cmat = [0 1 0.0;
    #             1 0 0]
    #     Ga = Gqa(RS.params(q_a),RS.params(q_b),vertices, joint_direction,cmat) 
    #     @time Gqa!(model, model.Gqamtx, RS.params(q_a),RS.params(q_b),vertices,cmat) 
    #     @test Array(Ga) ≈ Array(model.Gqamtx) atol=1e-9
    #     Gb = Gqb(RS.params(q_a),RS.params(q_b),vertices, joint_direction,cmat) 
    #     @time Gqb!(model, model.Gqbmtx, RS.params(q_a),RS.params(q_b),vertices,cmat) 
    #     @test Array(Gb) ≈ Array(model.Gqbmtx) atol=1e-9
    # end


    # # test state_diff_jac
    # @testset "state_diff_jac" begin
    #     model = FloatingSpace()
    #     n,m = size(model)
    #     n̄ = state_diff_size(model)
    #     @show n
    #     @show n̄

    #     x0 = generate_config(model, [2.0;2.0;1.0;pi/2], [pi/2]);
    #     @time attG = state_diff_jac(model, x0)
    #     @time state_diff_jac!(model,x0)
    #     @test attG ≈ model.attiG atol=1e-9
    # end

    # # test inplace derivatives
    # @testset "G derivative" begin
    #     model = FloatingSpace(4)
    #     q_a = UnitQuaternion(RotX(0.03))
    #     q_b = UnitQuaternion(RotY(0.03))
    #     vertices = [1,2,3,4,5,6]
    #     joint_direction = [0,0,1]
    #     cmat = [0 1 0.0;
    #             1 0 0]
    #     @time Ga = Gqa(RS.params(q_a),RS.params(q_b),vertices, joint_direction,cmat) 
    #     @time Gqa!(model, model.Gqamtx, RS.params(q_a),RS.params(q_b),vertices,cmat) 
    #     @test Ga ≈ model.Gqamtx atol=1e-9
    #     @time Gb = Gqb(RS.params(q_a),RS.params(q_b),vertices, joint_direction,cmat) 
    #     @time Gqb!(model, model.Gqbmtx, RS.params(q_a),RS.params(q_b),vertices,cmat) 
    #     @test Gb ≈ model.Gqbmtx atol=1e-9

    #     λt = [1;2;3;4;5.0]
    #     @time kk = ∂Gqaᵀλ∂qa(model, RS.params(q_a),RS.params(q_b),λt,vertices, joint_direction,cmat)
    #     @time ∂Gqaᵀλ∂qa!(model, model.dGqaTλdqa, RS.params(q_a),RS.params(q_b), λt, vertices,cmat) 
    #     @test kk ≈ model.dGqaTλdqa atol=1e-9
    #     @time kk = ∂Gqbᵀλ∂qb(model, RS.params(q_a),RS.params(q_b),λt,vertices, joint_direction,cmat)
    #     @time ∂Gqbᵀλ∂qb!(model, model.dGqbTλdqb, RS.params(q_a),RS.params(q_b), λt, vertices,cmat) 
    #     @test kk ≈ model.dGqbTλdqb atol=1e-9
    #     @time kk = ∂Gqbᵀλ∂qa(model, RS.params(q_a),RS.params(q_b),λt,vertices, joint_direction,cmat)
    #     @time ∂Gqbᵀλ∂qa!(model, model.dGqbTλdqa, RS.params(q_a),RS.params(q_b), λt, vertices,cmat) 
    #     @test kk ≈ model.dGqbTλdqa atol=1e-9
    #     @time kk = ∂Gqaᵀλ∂qb(model, RS.params(q_a),RS.params(q_b),λt,vertices, joint_direction,cmat)
    #     @time ∂Gqaᵀλ∂qb!(model, model.dGqaTλdqb, RS.params(q_a),RS.params(q_b), λt, vertices,cmat) 
    #     @test kk ≈ model.dGqaTλdqb atol=1e-9
    # end


    # # test dynamics
    # @testset "dynamics" begin
    #     using Random
    #     Random.seed!(123)
    #     model = FloatingSpace(3)
    #     x0 = generate_config_with_rand_vel(model, [2.0;2.0;1.0;pi/4], fill.(pi/4,model.nb))
    #     dr = pi/14
    #     x1 = generate_config_with_rand_vel(model, [2.0;2.0;1.0;pi/4+dr], fill.(pi/4+dr,model.nb))
    #     # x0 = generate_config_with_rand_vel(model, [2.0;2.0;1.0;pi/4], [pi/4])
    #     # dr = pi/14
    #     # x1 = generate_config_with_rand_vel(model, [2.0;2.0;1.0;pi/4+dr], [pi/4+dr]);
    #     u = 2*randn(6+model.nb)
    #     du = 0.01*randn(6+model.nb)
    #     λ = randn(5*model.nb)
    #     dλ = 0.001*randn(5*model.nb)
    #     dxv = zeros(model.ns)
    #     dxv[(13*0).+(4:6)] = randn(3)
    #     dxv[(13*0).+(11:13)] = randn(3)
    #     dxv[(13*1).+(4:6)] = randn(3)
    #     dxv[(13*1).+(11:13)] = randn(3)
    #     dt = 0.01
    #     @time f1 = fdyn(model,x1, x0, u, λ, dt);
    #     @time fdyn!(model,x1, x0, u, λ, dt)
    #     @test Array(f1) ≈ Array(model.fdyn_vec) atol=1e-10
    #     f2 = fdyn(model,x1+dxv, x0+dxv, u+du, λ+dλ, dt)
    #     @time Dfmtx = Dfdyn(model,x1, x0, u, λ, dt);
    #     @time Dfdyn!(model,x1, x0, u, λ, dt)
    #     @test Array(Dfmtx) ≈ Array(model.Dfmtx) atol=1e-10

    #     @time attiG_mtx = fdyn_attiG(model,x1,x0)
    #     @time fdyn_attiG!(model,x1,x0)
    #     @test Array(attiG_mtx) ≈ Array(model.fdyn_attiG) atol=1e-10

    #     # compare with Forward diff
    #     # faug(z) = fdyn(model, z[1:model.ns], z[model.ns+1:model.ns*2], z[model.ns*2+1:model.ns*2+6+model.nb], z[model.ns*2+6+model.nb+1:end], dt)
    #     # Df2 = ForwardDiff.jacobian(faug,[x1;x0;u;λ])

    #     # @test Dfmtx ≈ Df2
    # end


# test dynamics simulation
function test_dyn()
    model = FloatingSpace(1)
    n,m = size(model)
    n̄ = state_diff_size(model)
    @show n
    @show n̄
    x0 = generate_config_with_rand_vel(model, [2.0;2.0;1.0;pi/4], fill.(pi/4,model.nb))

    U = 0.01*rand(6+model.nb)
    dt = 0.001;
    λ_init = zeros(5*model.nb)
    λ = λ_init
    x = x0
    @time x1, λ = discrete_dynamics(model,x, U, λ, dt)
    @show fdyn(model,x1, x, U, λ, dt)
    # println(norm(fdyn(model,x1, x, u, λ, dt)))
    x = x0;
    for i=1:5
        println("step: ",i)
        @time x1, λ = discrete_dynamics(model,x, U, λ, dt)
        println(norm(fdyn(model,x1, x, U, λ, dt)))
        println(norm(g(model,x1)))
        x = x1
    end
end

function test2()
    model = FloatingSpaceOrth(4)
    x0 = generate_config(model, [0.0;0.0;1.0;pi/2], fill.(pi/4,model.nb));
    Tf = 25
    dt = 0.005
    N = Int(Tf/dt)

    # mech = vis_mech_generation(model)
    x = x0
    λ_init = zeros(5*model.nb)
    λ = λ_init
    U = 0.3*rand(6+model.nb)
    # U[7] = 0.0001
    # steps = Base.OneTo(Int(N))
    # storage = CD.Storage{Float64}(steps,length(mech.bodies))
    println("start to simulate")
    @time begin
        for idx = 1:N
            # println("step: ",idx)
            x1, λ1 = discrete_dynamics(model,x, U, λ, dt)
            # println(norm(fdyn(model,x1, x, U, λ1, dt)))
            # println(norm(g(model,x1)))
            # setStates!(model,mech,x1)
            # for i=1:model.nb+1
            #     storage.x[i][idx] = mech.bodies[i].state.xc
            #     storage.v[i][idx] = mech.bodies[i].state.vc
            #     storage.q[i][idx] = mech.bodies[i].state.qc
            #     storage.ω[i][idx] = mech.bodies[i].state.ωc
            # end
            x = x1
            λ = λ1
        end 
    end
end
