# matlab-for-robotics

## Misc
- Cubic_rest_to_move_PLOT: trajectory planning for a rest to move with general time $t_i$ and $t_f$, **with plots for $q(t)$, $\dot{q}(t)$ and $\ddot{q}(t)$** 
- Cubic_Spline_PLOT: 2 cubic for an intermediate point
- NewtonMethod: calculate $q*$ with the Newton Method
- GradientMethod: calculate $q*$ with the Gradient Method
- Inverse_kinematics_planar_2R: calculate $q$ from $r(q)$ for a planar 2R robot
- Inverse_kinematics_planar_3R: calculate $q$ from $r(q)$ for a planar 3R robot
- Inverse_kinematics_spatial_RRP: calculate $q$ from $r(q)$ for a spatial RRP robot
- Inverse_kinematics_planar_PR: calculate $q$ from $p(q)$ for a spatial PR robot
- Inverse_kinematics_planar_RP: calculate $q$ from $p(q)$ for a spatial RP robot
- Solution_direct_problem_rotation: find $R$ from $r$ and $\theta$ 
- Solution_inverse_problem_rotation: find $r$ from $R$
- TimeDerivativeJacobian: calculate $\dot{J}$
- TrajectoryPlanningAxisAngle 
- TrajectoryPlanningEul
- PieceWise_PLOT: plot a piecewise function 
- Workspace_planar_2R: plot the workspace of a 2R robot
- Workspace_spatial_3R: plot the workspace of a 3R robot 
- CalculateMinorsDeterminants: I guess it's self-explanatory
- Geometric_jacobian: compute the geometric jacobian from the DH table
- Control_law_exercises: a set of exercises in which I used control laws for having $e \to 0$
- Sensors: a set of exercises about sensors

## January 2024 (2024-01)
- Ex 4: Geometric Jacobian, rotation of $^3p_D$ using $^0A_3$ and not only $^0R_3$ since you need to take into account the traslation too 
- Ex 5: direct kinematics = trajectory $p_{d,x}(t)$, $p_{d,y}(t)$ and $alpha_d(t)$ to find $q_d(t)$. Confront between $r_{direct kin}(q_d(\bar{t})) = r_d(\bar{t})$ 
- Ex 6: trajectory in $T = t_f - t_i$ and constraint on $|\dot{q}(t)|$ 

## June 2023 (2023-06)
- Ex 2: planning with circular path, constraint on $||\ddot{p}||$
- Ex 3: geometric jacobian, mid frame to calculate the determinant

## February 2023 (2023-02)
- Ex 3: Newton iterative method
- Ex 4: geometric jacobian, range space, null space, complementary theorem

## January 2023 (2023-01)
- Ex 3: algebraic transformation
- Ex 4: range and null space, complementary theorem
- Ex 5: planning with helical path, contraint on $||\dot{p}||$, $|\ddot{p}^{T}t|$ and $|\ddot{p}^{T}n|$ 

## October 2022 (2022-10)
- Ex 1b: Jacobian, singularities, null and range space
- Ex 1c: Joint velocity control law **without the need of planning a trajectory** (so $p_d$ is constant and $\dot{p}_ = 0d$) to move from $p(0)$ to $p_d$
- Ex 2: bang bang profile for two joints, scaling of the faster joint imposing $T_1* = T_2*$

## June 2022 (2022-06)
- Ex 2: extracting $\ddot{q}$ from $\ddot{p} = J(q)\ddot{q} + \dot{J(q)}\dot{q}$, calculation of $\dot{J}$ the derivative of the jacobian
- Ex 3: balancing forces and torques
- Ex 4: cartesian kinematic control to get $e_p(t) = e^{-K_pt}e_p(0)$
  
## February 2022 (2022-02)
- Ex 3: $\ddot{q}$ from $\ddot{p} = J(q)\ddot{q} + \dot{J(q)}\dot{q}$ and $\ddot{e_p}$
- Ex 4: planning with an obstacle -> intermediate point, splines $q(\tau)$
- Ex 5: gears and pulleys reduction ratio, time $T_{\theta}$ *(not done any code for this)*

## January 2022 (2022-01)
- Ex 2: RPR inverse kinematics
- Ex 3: null and range space, pseudoinverse of the jacobian in a singularity
- Ex 4: $\tau = -J^{T}F$ and balancing forces and torques
- Ex 5: elliptic path and constraint on $||\dot{p}||$ and $||\ddot{p}||$ *(not done any code for this)* 
- Ex 6: singularities of a 2R robot, initial configuration $q_{N}(0)$ with zero cartesian error and $q(0)$ with $e(0) \neq 0$ 
- Ex 7: absolute encoder + bit for turns

## January 2021 (2021-01)
- Ex 5: absolute encoder, resolution link side, maximum angular displacement measureable by the encoder, Gray Code to motor angle
- Ex 6: jacobian, null and range space
- Ex 7: two robots pushing each other with $||F|| = 10\ [N]$ must remain in equilibrium
- Ex 8: 2R meeting a target at constant speed, intersection between line and circumference, $\dot{q} = diffinvkin(v)$
- Ex 9: $\ddot{q_1} = 0$ to find $T_peak$ that maximizes $\dot{q_1}$ and so on
- Ex 10: control law in tangent $e_t$ and normal directions $e_n$

## June 2020 (2020-06)

- Ex 2: jacobian singularities, $\tau$ for balancing $F$ applied to the end-effector in different configurations (singular and not) 
- Ex 3: range and null space for $J(q)$ for different ranks

## July 2022 (2022-07)
- Ex 3: Newton method and how to obtain another inverse kinematics solution (changing the initial guess)
- Ex 4: trajectory planning with continuity on acceleration -> quintic!

## September 2022 (2022-09)
- Ex 3: null and range space, **unique** solution of $\ddot{q}$ from $\ddot{r} = J(q)\ddot{q} + \dot{J(q)}\dot{q}$ 
- Ex 4: rest-to-rest motion -> the optimal is bang-coast-bang, state to rest motion

## September 2020 (2020-09)
- Ex 1: $\dot{R}(t) = S(\omega)R$ and $\dot{\omega}$

## January 2019 (2019-01)
- Ex 3: bang-bang acceleration as the time-optimal motion when not having constraint on $\dot{q}$ but only on $\ddot{q}$, uniform scaling

## February 2021 (2021-02)
- Ex 5: Newton and Gradient method for $\epsilon \to 0$ and $e \in N(J^T)$
- Ex 6: $J_A$ and $\dot{n} = S(w)*\dot{q}$ = $\omega \times n$ = $(J_A * \dot{q}) \times n$ 
- Ex 7: trajectory planning for end-effector orientation
- Ex 8: 3R in 2D must stay fixed in $P_in$ changing from $q_in$ to $q_fin$ with $q_{3, fin}$ fixed, 2D reduced robot. Joint space decomposition.

## June 2021 (2021-06)
- Ex 2: RPR robot for tube with vertical costant speed (task a) and static balance (task b) 
- Ex 3: linear path trajectory with simultaneous change of orientation

## September 2021 (2021-10)
- Ex 1: PRR, range, null and complementary of the range space. Inverse kinematics of a PRR. Primary and secondary workspace of a PRR. 
