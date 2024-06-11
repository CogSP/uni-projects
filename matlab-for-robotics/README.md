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

## February 2024 (2024-02)
- Ex 1: two cameras, perspective relationship between the camera frame and the image plane
	- inverse kinematics of a planar PPR
- Ex 2: RPP spatial cylindrical robot, DH, inverse kinematics, workspace
	- physical intepretation of $N(J_s)$ and $N(J{_s}^T)$
- Ex 3: cubic in joint space $q(s)$ and cubic timing law $s(t)$
	- split in space and time necessary since there are constraints on tangent direction $\frac{dp}{ds}$
	- find new T in order to find the uniform time scaling constant k in order to respect the constraint on $\dot{q}(t)$	
- Ex 4: workspace of a 3R robot = sphere + control law with error in the frenet frame
	- check if the helix is inside the sphere:
		1. the distance center-lowest_elix_point and center-highest_helix point must be less than the radius
		2. the radius of the helix must be less than the radius of the sphere

## January 2024 (2024-01)
- Ex 1: fixed axes ZXY
	- express an orientation in the frame with partial rotation ZX
- Ex 2: spatial 6R Yaskawa Motoman GP7 DH Table
- Ex 3: algebraic transformation
- Ex 4: Geometric Jacobian, rotation of $^3p_D$ using $^0A_3$ and not only $^0R_3$ since you need to take into account the traslation too 
- Ex 5: inverse kinematics from task trajectory $p_{d,x}(t)$, $p_{d,y}(t)$ and $alpha_d(t)$ to joint trajectory $q_d(t)$. Confront between $r(q_d(\bar{t})) = r_d(\bar{t})$ 
- Ex 6: joint rest-to-move trajectory $q(t)$ with generic initial time $t_i$
	- note that the maximum can also be the velocity at the end: $v_f$

## November 2023 (2023-11)
- Ex 1: Euler XYZ = partial Euler ZY
- Ex 2: axis angle method and fixed angles
- Ex 3: planar 2R with limits on the joint: banana workspace
- Ex 4: unusual DH for a planar 2R
- Ex 5: differential equation of a DC motor, eigenvalues to find stability of the system
- Ex 6: spatial 5R robot DH table
- Ex 7: planar RPR, direct and inverse kinematics
	- interaction between planar RPR and planar 2R: coincident end effectors 

## June 2023 (2023-06)
- Ex 1: spatial 6R ABB CBR 15000 robot, DH Table
- Ex 2: circular path and rest-to-rest timing law, analysis on the norm of the acceleration, trying to minimize $\sqrt(\alpha(\tau))$
- Ex 3: geometric jacobian, mid frame to calculate the determinant

## February 2023 (2023-02)
- Ex 1: planar RPPR robot, DH and maximum distance of the e.e. under prismatic constraints
- Ex 2: $e_{\alpha} = \alpha_{cd}$ is false since extraction of a minimal representation is non-linear
- Ex 3: Newton iterative method
- Ex 4: geometric jacobian, range space, null space, complementary theorem

## January 2023 (2023-01)
- Ex 1: TIAGo 8-DoF, P7R
- Ex 2: find $w^p_W
- Ex 3: algebraic transformation
- Ex 4: planar RPRP robot dh table, range and null space, complementary theorem
- Ex 5: planning with helical path, contraint on $||\dot{p}||$, $|\ddot{p}^{T}t|$ and $|\ddot{p}^{T}n|$ in the frenet frame
	- find the limits $v_max$ and $a_max$ of the bang-coast-bang $s(t)$ thanks to the constraint on the norms
	- good placement of the spatial 3R robot base s.t. the helical path is in its primary workspace without singularities

## November 2022 (2022-11)
- Ex 1: inverse axis-angle problem
- Ex 2: YXY Euler angles
- Ex 3: harmonic drive and gear
	- Direction of rotation: both the harmonic and the gear invert the rotation
	- Relationship between linear and angular variation $\delta r / L = \delta \theta$
- Ex 4: 6R robot holding a cylindric in front of the camera: task kinematic identity
	- positioning and pointing in 3D requires 5 DoF so the robot is redundant
- Ex 5: spatial RPR robot DH table
- Ex 6: RPR robot inverse kinematics 
	- 4 solutions in the regular case and without joint limit

## October 2022 (2022-10)
- Ex 1a: spatial RPR robot DH table
- Ex 1b: Jacobian and singularities, null and range space of the RPR robot
- Ex 1c: Joint velocity control law **without the need of planning a trajectory** (so $p_d$ is constant and $\dot{p}_ = 0d$) to move from $p(0)$ to $p_d$
- Ex 2: bang bang profile for two joints, scaling of the faster joint imposing $T_1* = T_2*$

## September 2022 (2022-09)
- Ex 1: 6R Fanuc cr15ia, DH Table
- Ex 2: *questionnaire*
- Ex 3: null and range space
	- obtaining $\ddot{r} = 0$ by commanding $\ddot{q} \neq 0$ when $\dot{q} = 0$ only if J is singular
	- solution of $\ddot{q}$ from $\ddot{r} = J(q)\ddot{q} + \dot{J(q)}\dot{q}$ is **unique** when the jacobian is not singular, since the null space is empty, otherwise you have infinite solutions
- Ex 4: joint trajectory $q(t)$
	- rest-to-rest with bang-coast-bang
	- move-to-rest with an initial velocity in the wrong direction

## July 2022 (2022-07)
- Ex 1: fixed axes XZY, linear mapping $T(\phi)$ between angular velocity $\omega$ and velocity of the e.e. $\dot{\phi}$, 2 ways of doing it
	- from $S(\omega$)
	- with a rule explained in #Differential Kinematics pag. 11
- Ex 2: spatial 3R robot inverse kinematics
	- fixing q in $l_2(q)$, we have a 2R robot
- Ex 3: Newton method and how to obtain another inverse kinematics solution (changing the initial guess, now hopefully closer to a different solution)
- Ex 4: trajectory planning with continuity on acceleration *on the whole interval* -> quintic!
	- is quintic since it's asking continuity in the whole interval, comprising the boundaries t = 0 and t = T, without the boundaries it would have been a cubic

## June 2022 (2022-06)
- Ex 1: spatial 3R, DH, workspace (torus), singularities
	- infinite solution for $\dot{q}$ in a singular configuration
- Ex 2: extracting $\ddot{q}$ from $\ddot{p} = J(q)\ddot{q} + \dot{J(q)}\dot{q}$, *unique* solution in a regular configuration
- Ex 3: balancing forces and torques
- Ex 4: control law with cartesian linear path
	- computation of the largest control gain that respect the bounds on $|\dot{q}|$
	-  time constant to discuss how fast the original trajectory will be achieved
  
## February 2022 (2022-02)
- Ex 1: 7R Crane-X7 robot DH table
- Ex 2: YXY Euler, $T_w = \frac{\theta}{\omega}$
- Ex 3: 3R planar robot command $\ddot{q}$ from $\ddot{p} = J(q)\ddot{q} + \dot{J(q)}\dot{q}$ and $\ddot{e_p}$ 
	- Modify the control law if there is a position or velocity error
	- Modify the control law if there is an constant angular velocity error
- Ex 4: planning with an obstacle -> intermediate point, splines $q(\tau)$
- Ex 5: gears and pulleys reduction ratio, time $T_{\theta}$ 
	- the gears inverts the rotation while the pulley perserves it

## January 2022 (2022-01)
- Ex 1: planar RPR DH
	- geometric reasoning with atan to find $3^T_e$
	- XYX Euler
- Ex 2: planar RPR inverse kinematics
- Ex 3: planar RPR singularities , null and range space
	- infinity of solution but can't find the one with minumum norm since we are adding apples and oranges confronting the norms7
	- usage of mldivide in the matlab code
- Ex 4: $\tau = -J^{T}F$ and balancing forces and torques
	- rotate the forces from $RF_e$ to $RF_0$
- Ex 5: elliptic path with constraint on $||\dot{p}||$ and $||\ddot{p}||$ 
- Ex 6: planar 2R inverse kinematics
	- condition on $l_1 = a$ and $l_2 = b$ to not meet singularities
	- control law with error only on the x component
- Ex 7: multi-turn absolute encoder

## November 2021 (2021-11)
- Ex 1: orientation of a body
- Ex 2: ZYZ Euler 
- Ex 3: spatial 4R DH Table
- Ex 4: banana workspace of a 2R robot
- Ex 5: 5R branched two-arm planar robot
	- Note that the angles $\theta$ are not the usual of the DH convention	 
- Ex 6: RRP robot inverse kinematics
- Ex 7: gears
	- no inversion of rotation (in the "planar" case there is inversion, e.g. ex 5 02-2022)
- Ex 8: absolute encoder
- Features of a SCARA-type robot

## September 2021 (2021-10)
- Ex 1: planar PRR direct, inverse, differential, primary and secondary workspace
- Ex 2: rest-to-rest joint trajectory $q_d(t)$ without changing $p_d$ but only $\alpha_d$

## July 2021 (2021-07)
- Ex 1: spatial RRRP DH table and singularities
- Ex 2: encoder and gearboxes
- Ex 3: rest-to-rest angular spline with two terms
	- ZYX Euler

## June 2021 (2021-06)
- Ex 1: 3P3R robot with a portal structure, DH table 
- Ex 2: RPR robot for tube with vertical costant speed (task a) and static balance (task b)
	- Note that the force is $f_x$, $f_y$, $\mu_z$, excluding the other component 
- Ex 3: linear path trajectory with simultaneous change of orientation

## February 2021 (2021-02)
- Ex 1: XYZ fixed axes
- Ex 2: planar 5R two-jaw robot DH Table
	- midpoint of the two jaws
	- distance between the two tips
	- angle of the left jaw w.r.t to the right
	- orientation angle w.r.t. to $x_0$ of the jaw pair
- Ex 3: incremental encoder with angular errors, questions
- Ex 4: planar PPR robot primary and secondary workspace 
- Ex 5: Newton and Gradient method for $\epsilon \to 0$ and $e \in N(J^T)$
- Ex 6: 3R pointing device. $J_A$ and $\dot{n} = S(w)*\dot{q}$ = $\omega \times n$ = $(J_A * \dot{q}) \times n$ 
- Ex 7: trajectory planning for end-effector orientation
- Ex 8: 3R in 2D must stay fixed in $P_in$ changing from $q_in$ to $q_fin$ with $q_{3, fin}$ fixed, 2D reduced robot. Joint space decomposition.

## January 2021 (2021-01)
- Ex 1: YXZ Euler
- Ex 2: angular velocity $\Omega$ expressed in the body
- Ex 3: matrix M as the inverse of the DH homogeneous transformation matrix A
- Ex 4: spatial 3R DH table 
- Ex 5: absolute encoder, resolution link side, maximum angular displacement measureable by the encoder, Gray Code to motor angle
- Ex 6: planar RRPR singularities 
- Ex 7: two robots pushing each other with $||F|| = 10\ [N]$ must remain in equilibrium
- Ex 8: 2R meeting a target at constant speed, intersection between line and circumference, $\dot{q} = diffinvkin(v)$
- Ex 9: $\ddot{q_1} = 0$ to find $T_peak$ that maximizes $\dot{q_1}$ and so on
- Ex 10: control law in the frenet frame

## November 2020 (2020-11)
- Ex 1: ZYX Fixed
- Ex 2: inverse axis-angle
- Ex 3: why do we need only 4 DH parameters and not 6?
- Ex 4: cost in terms of addition $N_+$ and products $N_x$
- Ex 5: we can move robot by commanding just $q$ or $\dot{q}$
- Ex 6: spatial PRPR robot DH table
	- orientation $R \in SO(3)$ that the robot can never assume: structural zero
- Ex 7: inverse kinematics of the spatial PRPR robot and primary workspace
- Ex 8: 1-step BDF Euler formula
- Ex 9: encoder and toothed belt
- Ex 10: just multiplication of homogeneous transformation

## September 2020 (2020-09)
- Ex 1: $\dot{R}(t) = S(\omega)R$ and $\dot{\omega}$, skew matrix
- Ex 2: 6R UR5 DH table
- Ex 3: a planar 2R and a planar 3R handing an object, task kinematic identity
- Ex 4: 3R spatial robot singular configuration where the rank is 2 and 1
- Ex 5: bang-bang state-to-rest for a mass with $M*\ddot{x} = u$ 

## July 2020 (2020-07)
- Ex 1: planar PRRR robot DH table
- Ex 2: planar PP cartesian robot eight-shaped periodic trajectory
- Ex 3: YXY EUler, mapping bertween $\omega$ and $\dot{\phi}$
- Ex 4: planar 3R linear cartesian path with e.e. orthogonal to the path
- Ex 5: *questionnaire*, Euler-RPY and inertia-link_velocity

## June 2020 (2020-06)
- Ex 1: spatial 4R DH table 
- Ex 2: planar RRP robot direct, differential (singularities) and statics 
- Ex 3: range and null space for $J(q)$ of a spatial 3R for different ranks
- Ex 4: joint cubic in time
- Ex 5: *questionnaire*, rotations, harmonic drive and encoder, 1-step and 4-step BDF 

## February 2020 (2020-02)
- Ex 1: spatial PRRR DH table 
- Ex 2: spatial PRRR robot direct, statics, differential
- Ex 3: planar PPR tracing a circle with the e.e. normal to the surface
	- $\dot{s} = \frac{v}{R}$
- Ex 4: *questionnaire*

## January 2020 (2020-01)
- Ex 1: spatial 7R CESAR robot DH table
- Ex 2: 7R CESAR robot differential of the wrist, inverse with Gradient method
	- infinite numnber of solution since it's a redundant robot
- Ex 3: RPR moving in contant with an oblique surface
	- $\alpha(s)$ w.r.t. to the normal to the surface
	- state-to-rest motion -> cubic (continuity up to acc in all the interval)
	- mapping the task into a joint motion, so do the inverse kinematics given:
		1. direct kinematics to match $p$ with $q$
		2. match geometrically the orientation of the task $\alpha(s)$ with e.e. orientation $\phi(s)$
- Ex 4: *questionnaire*

## November 2019 (2019-11)
- Ex 1: RPY, axis angle method
	- identity regarding rotation axis ${0}^r_{1}$ and ${i}^r_{1}$
- Ex 2: DH of the 6R UR10
- Ex 3: planar 2R workspace, DH and $2^R_e$ with atan
- Ex 4: Newton method
	- Note that the task kinematics is not using joint variables but $q_1$, $q_2$ and $q_3$ are representing other angles
	- Note that since $q3 - q2$ is a constraint, it mean that these two joints are equal and so the 3-DoF robot is actually a 2R and so there are two solution in the regular case
- Ex 5: *questionnaire*


## September 2019 (2019-09)
- Ex 1: determine $\ddot{q}$ s.t. $\ddot{p} = 0$. In a singularity $\dot{J}*\dot{q}$ must be in the range of the Jacobian, otherwise is not possible to determine $\ddot{q}$
- Ex 2: planar RP with a bang-coast-bang.
	- Note that the norm of the acceleration is computed before (-) and after (+) $\frac{T}{2}$ since it's discountinous
- Ex 3: laser scanner and quadrature incremental encoder. Lateral uncertainty $\delta L$


## July 2019 (2019-07)
- Ex 1: Jacobian of the payload ($J_e + J_ep$)
- Ex 2: planar RP, fit a line AB in its workspace and find RF_w from $A_w$, $B_w$, $A_0$, $B_0$
- Ex 3: bang-coast-bang for rotation "\delta \theta$


## June 2019 (2019-06)
- Ex 1: DH and Inverse of spatial 6R Kawasaki S030
	- Data sheet with max operating range and speed 
- Ex 2: pros and cons of expressing the linear velocity at the joint level, in the base frame and in the e.e/tool frame, commenting on the difference w.r.t. to $n > < = m$

## February 2019 (2019-02)
- Ex 1: DH of a spatial 4R
	- consequences of putting $d_1 = d_4 = 0$
- Ex 2: Differential kinematics of the spatial 4R
	- solve the determinant with:
		1. algebraic substitution
		2. one of the factor is $\sqrt(p_x^2 + p_y^2) \to p_x = p_y = 0$
	- redundant robot $\to$ infinite solutions for the inverse differential kinematics: add to the solution a vector of the null space
- Ex 3: Circular geometric path with error expressed in $RF_r(t)$
	- Since $0^R_r(t)$ depends on time, note that $e_r = 0^R_r(t)*\dot{e} + \dot{0^R_r(t)}*e so a *skew matrix* $S(\omega)$ is present
	- time constant $\tau = 1 / k$

## January 2019 (2019-01)
- Ex 1: DH of a spatial RRPR robot
- Ex 2: Range and Null space of the spatial RRPR robot
- Ex 3: bang-bang joint trajectory
	- there is no coast since $v_max$ not existing, so $L > \inf$
	- uniform time scaling to respect $||\ddot{p}|| <= A_c$
