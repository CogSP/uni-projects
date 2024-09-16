# matlab-for-robotics - Robotics 2

## Misc
- dynamic_model_planar: inputting T and the number of joints, compute
	- $M(q)$
	- $c(q, \dot{q})$
	- $g(q)$
- moving_frames_algorithm: find the dynamic model for spatial robot 
- fist_second_time_derivative_jacobian: computes the first and second time derivative of a Jacobian matrix
- get_minors: get all the minors of a matrix, useful to find the determinant by setting all minors to 0
- multiple_task: by inputting two tasks, if finds the extended jacobian an try to solve the problem, computing also the error
- bang-coast-bang-time-formulas: how to find $T*$ and $T_s$ in bang-coast-bang trajectories


## June 2010 (2010-06)
- RP Robot: dynamic model, redundancy (one-dimensional task) -> pseudoinverse
	- weight matrix $W$ to compare $\tau$ revolute joint torque and prismatic joint force
	- null space method to not build up joint velocities (TODO: check if this is actually the null space method)

## July 2010 (2010-07)
- mass/spring/damper scheme
	- differential equations of motion $F = ma$
	- Contact force control problem $F = k_f(F_d - F_c)$, determine $x_E$ and $e(x_e) = F_d - F_{c,e}$
	- Control action to remove the error
	- **TODO**: exponential stability with root locus analysis and Routh criterion 

## September 2010 (2010-09)
- RPR planar:
	- find $M$
	- minimal set of dynamic coefficient $a \in \mathcal{R}^p$ for $M$


## June 2011 (2011-06)
- 2R planar
	- robot hit by cartesian $F$: feedback control law $\tau$ s.t. $min\Vert\ddot{q}\Vert$
	- robot sensing requirements for achieving this result
	- control law $\tau$ for same behaviour but having only $q$ and $\dot{q}$

## September (2011-09)
- ex 1: 2R planar
	- find mechanical conditions s.t. it's self-balanced $\forall q$ w.r.t. gravity in absence of payload, i.e. $g(q) = 0$
- ex 2: application of PD + constant gravity compensation for $q_d$
	- now there is a point-wise payload: derive $g(q_d)$
	- find $M_{max}$ s.t. $(\ q_d\ \ \ 0\ )$ is global asymptotically stable: use $\Vert \nabla g(q) \Vert \leq \alpha$


## June 2012 (2012-06)
- Ex 1: visual servoing, pin-hole camera
	- interaction matrix $J_p = (\ J_v\ \ \ J_w\ )$ null space: camera motions that do not move the point feature in the image plane
	- $dim(\mathcal{N}(J_v\))$ to find which pure translation motion(s) $(v, 0)$ do not move the camera
	- $dim(\mathcal{N}(J_w\))$ to find which pure rotation motion(s) $(0, w)$ do not move the camera
- Ex 2: 3R planar
	- $F_c$ applied to the second link midpoint: generate $\tau_c$ only for $q_1$ and $q_2$. Still the third link accelerates due to the inertia coupling
	- Force becomes $2F_c$ but control law $\tau$ is the same: new equilibrium $\bar{q} \neq q_d$, so steady-state error
	- Using a force sensor you can update the control law when $F_c$ doubles, having 0 error


## July 2012 (2012-07)
- Portal Robot with 3 passive joints and 3 actuated joints (2 P and 1 R)
	- Dynamic Model
		- superposition of contribution $\dot{q}_1$, $\dot{q}_2$, $\dot{q}_3$ to $\omega$
		- usage of $\omega = \frac{v_{normal}sin(\theta)}{r}$
		- Reduced expression neglecting $(q_1 - q_2)^2$ that leads to $c(q, \dot{q} = 0$ 
	- PD + Constant Gravity Compensation Control Law: since $\Vert \nabla g \Vert = 0$, $K_p > 0$ guarantees asymptotic stability
		- exponential stability of a non-linear system using PD gains $BK_p$ and $BK_d$
	- show that $q = (\ q_1 \ \ q_2 \ \ q_3 \ )$ are generalized coordinates of the **closed kinematic chain** robot: the angular position and extension of the passive revolute and prismatic joints can be obtained from $q$


## January 2013 (2013-01)
- ex 1: 3R planar, two tasks
	- algorithmic singularities
	- task-priority strategy: **TODO**, the $\dot{q}$ formula used looks different from the one on the slides, are they equivalent?
- ex 2: RP robot move on a vertical line: **TODO** UNDERSTAND WHAT THIS IS ABOUT
	- constrained robot dynamics adding $A^T\lambda$


## February 2013 (2013-02)
- RP robot move on a vertical line: **TODO** SAME AS 2013-01 EX 1
	- this time the robot dynamics is **reduced** and not **constrained**: is it because we have a single first-order differential equation instead of a second order?


## July 2013 (2013-07)
- 3R robot, 3 scenarios with forces $F_i$ applied to links
	- identify which link is subject to $F_i$ thanks to $\tau_i = J_{c,i}F_i$: if from $\tau_i$ starts a 0 values sequence it means that the force is applied at link $i-1$
		- when $F_i$ is applied with $l_{c,i} = 0$ it is attributed to the previous link, i.e. $l_{c,i-1} = l_{i-1}$
	- knowing $l_{c,i}$ and $\tau_i$ find $F_i$: you can do if both $F_{i,x}$ and $F_{i,y}$ are present in $\tau_i = J_{c,i}F_i$. 
		- **TODO**: understand why in scenario 2, even if $J_2$ is not full rank, there seems to be a way of find $F_i$ at least partially
	- estimate $F_i$ without $l_{c,i}$: possible for third link
	- add the presence of gravity (nothing fancy about it)



## October 2014 (2014-10)
- ex 1: PRR planar
	- dynamic model subject to collision forces
	- residual formula
	- how the presence of prismatic and revolute joints can change the detection of collisions and contact forces
- ex 2: PRR planar
	- static condition, torques bounds $implies$ find maximum norm of a contact force $F$ that can be applied in any planar direction $\alpha$
	- note that a generic contact force can be parametrized as $F = \Vert F \Vert (\ cos(\alpha) \ sin(\alpha) \ )$



## April 2016 (2016-04)
- ex 1: PRR Robot: 
	- dynamic model
- ex 2: 4R planar
	- find $g(q)$
	- equilibrium configuration $q_e$ s.t. $g(q_e) = 0$
	- linear parametrization $g(q) = Y_G(q)a_G$
	- $d_i$ s.t. $g(q) = 0 \forall q$, i.e. $a_G = 0$
- ex 3: 4R planar
	- Projected Gradient (PG) to execute $r(t)$ while increasing $H_{range}(q)$


## June 2016 - Final Test with Midterm (2016-06-midterm)
- 2R robots with $h(q) = q1 + q2 - \beta = 0$
	- reduced dynamic model: $A$ constant
	- input to keep static equilibrium, i.e. $\dot{v} = 0$
		- note that $\lambda(0)$ should be zero if the constraint $h(q) = 0$ is **virtual**, i.e. is not imposed by mechanism but enforced through control action. $\lambda(0)$ should vanish since there can be no real constraint forces generated (so in this case we should find the condition on $u(0)$ that leads to $\lambda(0) = 0$).
	- **TODO**: simulation with $u(t)$, I don't know where to find this thing in the theory
	- compute the reduce dynamic for $\dot{v} = \dot{v}_d$ and $\lambda = 0$

## June 2016 (2016-06)
- ex 1: 2R polar robot
	- dynamic model
		- calculating $w_2$ expressed in $RF_0$ as sum of product between $\dot{q_i}$ and unit-norm vectors $z_i$ expressed in frame $RF_0$. This is taken from the angular part of the geometric Jacobian.
		- uniformly distributed mass and cylindric form: $I_{2x} \neq I_{2y}$ and $I_{2y} = I_{2z}$. I guess because in $y$ and $z$ direction the cylinder is symmetrical, while on the $x$ direction is built different.
	- adaptive control law, minimum dimension of the adaptive controller equal to the minimum number of dynamic coefficients
- ex 2: **TODO**


## July 2016 (2016-07)
- ex 1: 2R planar:
	- find $M$ with absolute coordinates
	- No Coriolis terms $c_{kij}$ with $i \neq j$ since we are using absolute coordinates
	- input $u$ for absolute coordinates vs $u_{\theta}$ for dh joint coordinates 
- ex 2: peg-in-hole frictionless task
	- natural vs virtual constraint
	- if there is no clearance you can put all virtual constraint to zero besides $v_z = v_{z,d} > 0$. Otherwise, if a firm contact needs to be maintained with one side of the hole you can choose $F_{x,d}$ and/or $F_{y,d}$
- ex 3: peg-in-hole compliant behaviour
	- **TODO**


## September 2016 (2016-09)
- ex 1: RPR planar:
	- dynamic model
	- equilibrium configuration $g(q) = 0$
- ex 2: two-mass and a non-linear spring system
	- dynamic model: $U = U_{gravity} + U_{elastic}
	- equilibrium position
	- cardano's formula for singe real solution of depressed cubic equation


## October 2016 (2016-10)
- ex 1: RP planar:
	- dynamic model with uniform mass distribution
	- equilibrium configuration under $u = 0$, i.e. $q$ s.t. $g(q) = 0$
- ex 2: 3-masses-2-springs
	- dynamic model using Lagrange
	- equilibrium states: $\ddot{q} = 0$
		- unforced state u = 0 $\to \dot{q} = 0$
		- forced state with constant input force $\bar{u}$ $\to \dot{q} equal for all masses$
	- prove with Lyapunov/linearity of the system that the proportional controller $u = k_p(q_d - q_1)$ asymptotically stabilizes the system to a unique equilibrium state (I GUESS IT'S $(q_d, 0)$) **TODO: TO UNDERSTAND BETTER**


## March 2017 (2017-03)
- ex 1: 3R spatial robot
	- determine $M(q)$ using moving frames
		- **TODO**: check if the calculations are okay
	- minimal number of dynamic coefficients
- ex 2: Jacobian weighted pseudoinverse
	- prove the general form, valid when the Jacobian loses rank too
	- **TODO**
- ex 3: 4R planar:
	- SNS method to find  $\dot{q}$ with minimum norm that respects the limits on the joints
	- **TODO**: understand we didn't had to scale, but when do we need it? When after applying the algorithm and saturating joints one by one we find at the last iteration that the last joint overcome the bounds?
- ex 4: 3R planar:
	- Reduced Gradient (RG)
		- choose the minor with largest determinant
	- Task Augmentation: second link endpoint on a circle
		- algorithmic singularities
- ex 5: 2R planar
	- calibration: find the regressor matrix
	- **TODO**: understand how you can find $\delta a$ and $\delta \theta$, are they the unknowns? Maybe the calibration algorithm aims at finding them.


## May 2017 (2017-05)
- Ex 1: given the lagrangian model of a n-joints manipulator
	- list all feedback control laws for $\tau$ that allows regulation to $q_d$
	- when PD achieve and does not achieve asymptotic stabilization
	- **TODO**: there are some problems if you ask me:
		- First: it is saying that the PD controller achieves asymptotic stabilization also in the presence of gravity (?), but on the slides it is written that $g(q) = 0$ is a condition, and I think for each $q$, not only $q_d$.	
		- Second: it is saying that these are only sufficient condition, not necessary in general, where is this written?
- Ex 2: visual servoing
	- from polar to cartesian coordinates
	- decoupled effects on $u$ and $v$
	- DLS control law when the point P gets close to the optical axis $\rho \to 0$, i.e. trespassing a singularity
- Ex 3: PRP planar
	- find $g(q) = Y_{g} a_{g}$
		- note that $g_0$ is not part of the dynamic coefficient
	- adaptive control
		- when both $a_M$ and $a_G$ are unknown
		- when $a_M$ is known
- ex 4: polishing task frame and natural and artificial constraints
	- point-wise area of contact means momentum $M = 0$

## June 2017 (2017-06)
- ex 1: 3R planar
	- differential inversion scheme at jerk level $\dddot{q}$ to follow desired trajectory $p_d(t)$, minimizing norm
	- modify command law $\dddot{q}$ s.t. $e$, $\dot{e}$, $\ddot{e}$ converges to zero
		- Hurwitz polynomial $k(s) = s^3 + k_2 s^2 + k_1 s + k_0$
- ex 2: 3R planar
	- PD with gravity compensation: largest value of $k_p$ s.t. $\tau(0) \leq T_{max}$
	- Transformation from $q$ to $\theta$ and Principle of Virtual Work (Power actually, since $\tau q = \tau_{\theta} \theta$
- ex 3: PRP planar
	- dynamic model
	- find a factorization $c = C\dot{q}$ s.t. $\dot{M} - 2C$ is skew-symmetric
	- find $\alpha$ and $\beta$ s.t. $\Vert \nabla g(q) \Vert \leq \alpha + \beta |q_3|$

## July 2017 (2017-07)
- ex 1: RPP cylindrical spatial
	- dynamic model and DH frames
- ex 2: PP planar
	- reduced dynamics: linear surface $y = mx + q$
	- $u = (\ u1 \ \ u2\ )$ s.t. $\dot{v} = 0$ and $\lambda = 0$
		- system of 2 equations: dynamic model and $\lambda$ equation
- ex 3: 3R planar
	- equilibrium in contact with a rigid obstacle: $\tau$ has a component $\tau_e = J(q_e)^{T}K_{p}(p_d - p(q)) = -\tau_c$ that compensate for the contact force of the object
	- momentum-based residual **TO UNDERSTAND**
	- **last question**


## September 2017 (2017-09)
- ex 1: RP planar
	- dynamic model: $M$, $c$ and $g$
	- plane $(x_0, y_0)$ inclined by $\alpha$ w.r.t. horizontal plane around $x_0$
	- $\ddot{q}_0$ s.t. robot at rest, i.e. $\dot{q}_0 = 0$ and $u = (\tau, F)$
	- $\Vert\ddot{p_0}\Vert^2$ and $q_{min}^{\*}$ and $q_{max}^{\*}$, given bounds on $\tau$, $F$ and $q_2$ 	
		- how $min\Vert\ddot{p}_0\Vert^2$ and $max\Vert\ddot{p}_0\Vert^2$ change if $\frac{I_1}{I_2}$ changes
- ex 2: state diagram in collision-aware tasks
	- $x_d(t) = (\ p_d(t) \ \phi_d(t)\ )$ cartesian linearization law
	- redundancy for cartesian task

## January 2018 (2018-01)
- ex 1: RP planar
	- $y_d(t)$ task	
		- minimize $\Vert \dot{q} \Vert^2$ with pseudoinverse: problem is inconsistency of measurement unit
		- weighted norm
		- inertia matrix weighted norm, minimize kinetic energy
- ex 2: Boulton-Watt governor
	- nonlinear feedback law $\tau_{\Omega}$
- ex 3: **TODO**


## February 2018 (2018-02)
- ex 1: $\eta = \Vert e \Vert$ task and its Jacobian $J_{\eta}$ expressed w.r.t to visual servoing $J = J_p J_m$
- ex 2: PRP cylindrical spatial mounted on a vertical wall
	- adaptive control law
- ex 3: iterative learning with pendulum equation
	- proof used to find the number of iteration needed to have $\epsilon < 0.01$


## March 2018 (2018-03)
- ex 1: automated crane
	- dynamic model
	- linear parametrization of the model with dynamic coefficients
	- linear approximation of the nonlinear model for small variation around $x_0 = (\ q_1\ q_2\ \dot{q}_1\ \dot{q}_2\ ) = **0**$
	- nonlinear state feedback law $F = F(**x**, a = \ddot{q}_1)$
- ex 2: **TODO**


## April 2018 (2018-04)
- ex 1: planar PPRR
	- dynamic model with viscous friction
	- linear parametrization $Y(q, \dot{q}, \ddot{q})a = u$
- ex 2: given 2-dof robot inertia matrix
	- find $S_1(q, \dot{q})$ and $S_2(q, \dot{q})$ s.t. $\dot{B} - 2S_1$ is skew-symmetric and with $S_2$ not
	- cubic trajectory for $q_2$ and kept steady $q_1$
		- $\tau(0)$ has same value on both component: since there is **inertial coupling** (values $m_ij \neq 0$ outside of the principal diagonal), $\tau_1(0)$ need to compensate the movement of the second joint due to $\tau_2(0)$
- ex 3: **TODO** a theoretical proof
- ex 4: PPR planar
	- pseudoinverse and weighted pseudoinverse

## June 2018 (2018-06)
- ex 1: RP planar
	- dynamic model
	- $S_1$ s.t. $\dot{M} - 2S_1$ is skew-symmetric
	- residuals **TODO: THIS DEFINITION IS DIFFERENT FROM THE ONE ON THE SLIDES**  
	- i-th collision: $det(J_{Ki}) = 0$ and $J_{Ki}^T*F_{Ki} = 0$
- ex 2: adaptive control law for $i_m$ of actuated pendulum
	- **known** length l means that is outside of the dynamic coefficients
- ex 3: mass-spring-damper system
	- class of control laws: $F = \alpha * k_f * (F_d - F_c) + \beta * F_d$
	- equilibrium point and asymptotic stability proven with Lyapunov
	- robustness w.r.t. to $m$, $d$ and $k_s$ 
	- what happens when %F_c = 0$, do we reach a steady state ($\ddot{x} = 0)? Yes, steady state velocity $\dot{x} = F / c$

## July 2018 (2018-07)
- ex 1: 2R robot with motors mounted on the axes of the joints
	- Inertia Matrix $M(q)$ with $q = (\ \theta^T \ \theta_m^T \ )$
- ex 2: SNS method for acceleration $q_ddot$
	- SNS algorithm solution has the least possible norm
- ex 3: visual servoing scheme
	- average interaction matrix $\bar{J}$, that is different from the interaction matrix of the average features parameters $(\ \bar{u} \ \bar{v}\ )$ and $\bar{Z}$
- ex 4: reduced dynamic model after partition $q = (\ q_a \ q_b \ )$


## April 2019 (2019-04)
- ex 1: which inertia matrix is associated to a 2-dof robot
	- Inertia Matrix must be symmetric and positive definite
	- $M_A$ isn't since it has $q_1$ coordinates in the inertia matrix: DH coordinate $q_1$ is chosen arbitrarily so an intrinsic property of the manipulator structure (as the inertia matrix) can't depend on it
	- $M_B$ is the inertia matrix of a 2-dof parallelogram structure
	- $M_C$ has negative determinant: is not positive definite
	- $M_D$ is the inertia matrix of a 2P robot: you can tell by the fact that we don't have any inertia term $I_i$ in the matrix $M_D$
- ex 2: PRP planar
	- dynamic model
	- linear parametrization $Y(q, \dot{q}, \ddot{q})a = u$
- ex 3: nR planar, with $q$ absolute coordinates
	- generic expression of $g(q)$
	- equilibrium configurations ($g(q_e) = 0$)
	- condition on the center of mass ($d_{ci}$) s.t. ($g(q) = 0 \forall q$)
- ex 4: given $M(q)$ find $S$ s.t. $\dot{M} - 2S$ is skew-symmetric
	- $S' \neq S$ s.t. $\dot{M} - 2S'$ is skew-symmetric: $S' = S + S_0$ with $S_0$ s.t. $\dot{q} \times \dot{q} = S_0 * \dot{q}$
- ex 5: 2R planar Jacobian with uni-dimensional task $\Vert p(q) \Vert$
	- inertia-weighted pseudoinverse to minimize the kinetic energy
- ex 6: SNS method for acceleration $q_ddot$


## June 2019 (2019-06)
- ex 1: 6R spatial (Kawasaki S030)
	- find $g(q)$: use DH or guess that $g_1 = g_4 = g_5 = g_6 = 0$
	- $q_e$ s.t. $g(q_e) = 0$
- ex 2: check if $M(q)$ can be a 3-dof serial robot inertia matrix and which condition
	- necessary condition: positive diagonal elements
	- necessary and sufficient condition: Sylvester criterion
- ex 3: with Singular Value Decomposition (SVD) show
	- $\dot{x}_d$ and $\dot{x}$ make a relative angle that is smaller than $pi/2$. That means that $\dot{x}_d \cdot \dot{x} \geq 0$
	- $\dot{q}_A has no task error while $\dot{q}_B$ has $\dot{e} = \dot{x}_d - \dot{x} \neq 0$
- ex 4: feedback linearization control law in the Cartesian space
	- since $K_p$ and $K_d$ are diagonal, we want a decoupled dynamics. The simple cartesian PD regulation law is not ok
	- transient behaviour of the error $e(t) \to 0$ using the Laplace domain **TODO**
- ex 5: cube sliding along a path on a flat surface
	- natural and artificial constraint
	- how many control loops? Three on the force $(\ F_z, \ M_x, \ M_y\ )$ and three on the motion $(\ w_z, \ v_x, \ v_y\ )$
	- how many DoF needed? 6-dof
	- The SCARA satisfy four control specification (three revolute, one prismatic acting orthogonally, so $F_z = F_{z,d}$). If the joint axes are normal to the plane of motion of the cube we have $M_x = M_y = 0$ so everything is satisfied. 
	- The 3R planar satisfies $v_x$, $v_y$ and $w_z$ but can't satisfy the rest.


## July 2019 (2019-07)
- ex 1: 3R planar task priority (TP)
	- drawing for region of compatibility (not understood)
- ex 2: RP planar collision
	- impact force $F_c$ normal to the second link
- ex 3: actuated pendulum bang-coast-bang
	- minimum time $T$ s.t. $\tau \leq \tau_{max}$

## September 2019 (2019-09)
- ex 1: control law $\tau$ that satisfies $\frac{dT}{dt} = - \gamma T
	- we find that $\tau = - \frac{\gamma}{2} M \dot{q}$
- ex 2: RP planar rest-to-rest along a circular path
	- bang-coast-bang
- ex 3: elastic impact between two masses $m_1$ and $m_2$, with $v_{i,2} = 0$
	- what happens when $m_1 > m_2$, $m_1 = m_2$, $m_1 < m_2$, $m_2 \to 0$, $m_1 \to \inf$


## April 2020 (2020-04)
- ex 1: when and why choose a two-stage calibration procedure
	- when some parameters have larger error
- ex 2: calibration of a n-dof serial manipulator, find regressor matrix
	- usage of taylor series 
- ex 3: $\dot{q} = $J^{\verb|#|}$ \dot{x}$ solution to minimize norm and $\dot{q} = $J_W^{\verb|#|}$ \dot{x}$ with different weights (same weights is equal to having no weights)
	- both solutions return the correct cartesian velocity, but $\Vert \dot{q} \Vert$ is minimum in the first case 
- ex 4: Task Priority method: e.e. position and last link upwards
- ex 5: total energy E and Lagrangian L are the same when $U = 0$ and $\dot{E} = \dot{q}^T(t) u(t)$
- ex 6: 2R polar robot, find $S'$ and $S''$ for skew-symmetry
- ex 7: PPR planar
	- dynamic model
	- linear parametrization
- ex 8: transformation $f(q)$ from DH to $p = (\ x \ y \ \alpha \ )$
- ex 9: single link (pendulum) rest-to-rest swing-up maneuver with bang-bang acceleration
	- torque limit not satisfied: time scaling, inertial torque scales with $k^2$
- ex 10: Newton-Euler routine
	- compute kinetic energy by just one call and one scalar product


## June 2020 (2020-06)
- ex 1: 3R planar SNS algorithm on velocity
- ex 2: conditions on elements of $M$ and $g$ in order to have the dynamic model of a 2-dof robot
	- since $g_1$ is zero, the first joint is an horizontal prismatic joint
	- find $q_ddot$ from dynamic equations
- ex 3: planar RP robot, find $\alpha < K_{p,m}$ having limitation on joint $d \leq q_2 \leq L$
 - ex 4: PP cartesian planar with payload $m_p$
	- viscous friction at joints	
	- adaptive control law
- ex 5: natural and artificial constraints of a robot closing a door
- ex 6: 2R planar, collisions with $F_{c,i}$
	- energy-based method vs momentum-based method to find if the collisions are detected and in which link


## July 2020 (2020-07)
- ex 1: PRRR (3R mounted on a rail)
	- find $M(q)$
- ex 2: two tasks: cirle with the e.e. and keep second link horizontal
	- augmented jacobian
	- find determinant with minors method
	- find the first point $P_s$ on the circular path where there is an algorithmic singularity
	- although there is a singularity, there is no error on the tasks, since $\dot{r}_d \in$ Range of $J_A(q_s)$
- ex 3: natural and artificial constraint of a cylinder moving along a path in contact with a planar surface
	- $x_t$ direction of the path
- ex 4: 2R planar
	- uniform time scaling factor
	- computation of torque after scaling
- ex 5: feedback control laws for regulation
	- global **exponential** stabilization of $(\ q, \ \dot{q} \ ) = (\ \dot{q} \ 0 \ )$ having $e(t) = e(0)(1 + 5t)^{-5t}$. Obtained with Feedback Linearization $u = M a + c + g$ with $a$ simple PD. To get $K_p$ and $K_d$ you insert the given $e(t)$ in $\ddot{e} + K_D\dot{e} + K_P e = 0
	- global **asymptotic** stabilization of of $(\ q, \ \dot{q} \ ) = (\ q_d \ 0 \ )$ **not knowing the robot inertia matrix**. Obtained with PD + gravity compensation on $u$. 
	- **exponential** stabilization of the e.e. $p = p_d$ with $\dot{p} = 0$. Obtained with feedback linearization in the Cartesian Space
	- Basically if you need exponential stabilization you need feedback linearization, and if you don't have information on the inertia matrix you need PD on $u$, not $\ddot{q}$.
	- principle of polynomial identity to find $K_p$ and $K_d$


## September 2020 (2020-09)
- ex 1: sensor-based obstacle avoidance
	- clearance function $H(q) = min \Vert a(q) - b \Vert$ where the point $b$ of the obstacle is made by the proximity sensor on the end-effector
	- Projected Gradient: switching control law on $\dot{q}$ that when $H(q) \geq \epsilon$ gives priority to $\dot{r}$ trying to maximize $H(q)$, while $H(q) \leq \epsilon$ we focus on avoid the obstacle
		- Note that we have a control $K_p$ on the robot when executing the task $\dot{r}$ since the phase of obstacle avoidance ($H(q) \leq \epsilon$) can introduce an error on $\dot{r}$ exiting the task path
- ex 2: sensor-based
	- barycentre of the triangle
	- **Important Note**: Jacobian of the barycentre of the point is different from the Jacobian of the barycentre
	- null space of the jacobian ($\dot{b} = 0$)
- ex 3: RP robot
	- cartesian inertia matrix $M_p = J^{-T} M J^-1$
	- cartesian dynamic equation to find $\ddot{p} = M_p^{-1} F$. Since for the given value $M_p^{-1}$ is diagonal and with the same values for each element of the diagonal, we have that $\ddot{p} = \alpha F \implies$ \ddot{p}$ has the same direction of $F$ 
- ex 4: 3R planar constrained, guide on vertical direction at a distance k
	- find $A(q)$, $D(q)$, $E(q)$, $F(q)$
	- find reduced inertia matrix
- ex 5: 2R robot 
	- known, symbolic form of the inertia matrix of the 2R robot with dynamic coefficient
	- arm of human-like size and weight ($a_3 = I_2 + m_2 d_{c2}^2 \geq 1$) 
	- Feedback linearization control $u_{FBL}$ vs Lyapunov-based $U_{GLB}$ and $\delta u$
	- which between $u_{FBL}$ and $U_{GLB}$ uses the larger instantaneous torque



## January 2021 (2021-01)
- ex 1: 3R robot Projected Gradient using $H(q) = U_g(q)$
	- since it's a control law we need to recover from possible error: $\dot{q} = $J^{\verb|#|}(\dot{r} - K_p e) + P \nabla U$
- ex 2: 3R robot, now torque controlled
	- control law to keep $P_d$ and minimize $U(q)$ as in the previous exercise
	- Feedback linearization law $\tau = M(q)a + c + g$ where $\ddot{q}= a$ and a is found from $\ddot{r} = J * \ddot{q} + \dot{J} * \dot{q}$ adding the null-space term
- ex 3: 2R spatial polar robot
	- dynamic model
	- viscous friction
- ex 4: 2R spatial polar robot
	- equilibrium with minimum torque $\tau$
		- from the dynamic model we find $\bar{\tau}_2$, then we check the solution with $\bar{\tau}_1 = 0$ since it certainly has the minimum norm
	- minimum number of dynamic coefficient
		- trigonometric substitution $s_2^2 = 1 - c_2^2$: assuming $I_{2,xx} \neq I_{2,yy} = I_{2,zz}$, the number of parameters with or without this substitution are the same. However, if $I_{2,yy} \neq I_{2,zz}$ we will have 1 coefficient more without the substitution.
- ex 5: PPR planar, momentum-based residuals
	- detect collision: if the force is in the null space it can't be detected
	- isolate colliding link
	- identify colliding force $F_c$: for instance, if $\tau_i = (\ F_x \ 0)$ it means that only the intensity $F_x$ of the force can be identified
	- determine location of collision point: if $\tau_i = J_i^T*F_i$ depends on $\rho_i$ we can localize the point of collision
	

## February 2021 (2021-02)
- ex 1: RRPR planar 
	- find M, g and their dynamic coefficients
	- note that $q_2$ is not a DH coordinate
- ex 2: two PP cartesian planar robot cooperating to move a payload in a path and minimizing $H = \Vert \tau_A \Vert ^2 + \Vert \tau_B \Vert ^2$
	- torque limits $\implies$ bang-bang profile
- ex 3: 3R planar torque control task rejecting position and velocity error
	- self-motion keeping $P_in$ but getting to $q_3 = -\frac{pi}{2}$
	- joint motion $\ddot{q} = a$ on the linearization law on $\tau$ will move $q_3$, adding a PD for errors
	- joint space decomposition approach
	- **TO RECHECK, IT IS NOT COMPLETELY CLEAR**
- ex 4: damper, mass, damper and spring
	- dynamic model
	- regulation control law using Laplace domain


## April 2021 (2021-04)
- ex 1: 2R planar
	- dynamic model having $rc_i$ given, w.r.t to $RF_i$
	- linear parametrization
- ex 2: 3R planar with absolute coordinates, task of keeping P on a vertical line at constant speed and second link horizontal
	- find first algorithmic singularity $q_s$
		- $q_3 = q_1$ and $p_x = c_1 + c_2 + c_3 = c_2 + 2c_1 = constant$, since the second link is horizontal $2c_1 = constant$ from which we can find $q_1 = q_3$, the first singularity encountered
	- $\dot{q}_{PS}$, $\dot{q}_{DLS}$, $\dot{q}_{TP}$ and compare their errors
- ex 3: PR planar
	- dynamic model
	- rest-to-rest circular path $p(s)$ in minimum time with bound on torques: bang-bang


## June 2021 (2021-06)
- ex 1: prove that the weighted pseudoinverse can be computed as $J_W^{\verb|#|} = W^{-\frac{1}{2}}pinv(JW^{-\frac{1}{2}})
- ex 2: single link mounted on a passive elastic support
	- Lagrangian dynamic model
	- input torque $\tau_0 > 0$ when the robot is at rest: will the robot move CW or CCW? Will the spring compress or extend?
		- find $\ddot{q}_1$ from the first equation and substitute it in the second equation, finding $\tau$ depending only on $\ddot{q}_2$
	- control law $\tau$ commanding $q_2$, classical PD knowing the model
		- dynamic of an undamped mass suspended on a spring
- ex 3: draw the 3-dof robot given $M$ and $g$: it's a cartesian 3P robot with a portal structure
- ex 4: 3R planar with absolute coordinates
	- robot has motors producing torques $u = (u_1, u_2, u_3)$ working on DH coordinates $\theta$. Provide torques on absolute coordinates $u_q = (u_{q1}, u_{q2}, u_{q3})$
- ex 5: 2R planar circular path trajectory
	- critically damped


## July 2021 (2021-07)
- ex 1: 3R planar, controlled by $u = \ddot{q}$ with bound $U_{max,i}$
	- is it possible to define $u_0 = \ddot{q}_0$ s.t. $\ddot{p}_0 = 0$? 
		- we know for sure that $\ddot{p} = J*u + \dot{J}\dot{q} = J*u + h$, and so when $\ddot{p} = 0$ we have $J*u = -h$
		- **if $h \in R(J)$** we have that $u$ will be a feasible command giving us $\ddot{p}$, so we just need to check the bounds $U_{max,i}$. In the case the bounds are not satisfied we can apply SNS. **Note**: since there is only one degree of redundancy, we can solve the problem with SNS only if at most 1 joint is out of the bounds
			- Note that **if $h \in R(J)$** can be true also in a singularity, since the range of the jacobian may be reduced but not empty
		- **if $h \notin R(J)$ we have that $u$ is unfeasible, so we can't return $\ddot{p} = 0$ but a result with the minimum norm using the pseudoinverse
- ex 2: incipiend block fault
- ex 3: RPR planar 
	- adaptive trajectory tracking control law with partly unknown dynamic model
	- proof of the asymptotic stability of the trajectory tracking error


## September 2021 (2021-09)
- ex 1: 3R planar, controlled by $u = \ddot{q}$ with bound $U_{max,i}$
	- which feasible $u_0$ to stop as fast as possible the cartesian motion while keeping the velocity $\dot{p}$ aligned with $\dot{p}_0$
		- $\ddot{p}$ = -\lambda \dot{p}$ choosing largest $\lambda$ s.t. $u_0$ is inside the bounds $implies$ linear program (LP)
- ex 2: 3R planar, provide eigenvalues of the $2 \times 2$ cartesian matrix $M_p$
	- note that since the jacobian is non-square, you can't use $M_p = J^{-T} M J^{-1}$ but you need to use $M_p = (J M^{-1} J^T)^{-1}$, always assuming that the jacobian is full rank
- ex 3: cartesian 2P planar
	- design impedance control law s.t. both eigenvalues are negative and coincident
	- dynamic model in contact with an environment $M\ddot{q} + g = \tau + F$
	- decoupled impedance model $M_d \ddot{e} + D_d \dot{e} + K_d e = F$
	- cartesian robot means $p = (q_1, q_2)'$
	- no force/torque sensor: $M_d \to M$, so we have that $\tau = $M\ddot{q} + g - F = M\ddot{q} + g - M \ddot{e} + D_d \dot{e} + K_d e = M\ddot{p}_d + g - D_d\dot{e} - K_d e
	- using Laplace you can impose $\lambda < 0$


## January 2022 (2022-01)
- ex 1: RPR planar with payload $m_p$ with inertia $I_p$
	- weighted inertia pseudoinverse
	- circular obstacle $O_{obs}$ of radius $r$ in position $P_{obs}$. We use Projected Gradient (PG) with clearance
		- $b$ in the clearance is $b = p_obs + r \frac{a(q) - p_{obs}}{\Vert a(q) - p_obs \Vert}$ since the obstacle is circular
- ex 2: dynamic model of an $nR$ robot
	- define joint control law continuous w.r.t. time that brings the robot in T seconds to an equilibrium state: cubic joint trajectory 
	- minimum factor $k$ for uniform time scaling
		- since the control $u$ and its bounds are directly on $\ddot{q}$ we have the ratio between the acceleration $\ddot{q}$ that overcomed the bounds and the bounds itself. When $u$ is on the torque of the dynamic model the formula is different, since it consider $u_{inertia}$ and $u_{gravity}$
- ex 3: PR planar move in a line between A and B with a cubic profile in T
	- no constraint force generated during motion ($\lambda = 0$)
	- reduced dynamic model and its control law
	- inverse constrained dynamic control law: find $\tau$ from the reduced dynamic model
		- formula of $\ddot{q}$ using $F$, $\dot{v}$, $E$, etc.


## February 2022-02 (2022-02)
- ex 1: RPR planar
	- linear parametrization $g(q) = G(q)a_G$
	- control law $\tau$ driven by Cartesian error $\implies$ Cartesian Control Law: a simple PD on cartesian coordinates. You get $\tau = J^T K_p e_p - K_d \dot{q} + g$
	- find configuration $q_s$ s.t. $e_p = p_d - f(q_s) \neq 0$ but the robot doesn't move under the action of previous control law $\tau$: put $\tau = g$ to compensate the gravity and $\dot{q} = 0$, now you have $J^T K_p e_p = 0$, so $K_p e_p \in N(J^T)$ will give you $q_s$ 
- ex 2: 4R planar
	- e.e. trajectory $p_d(t)$ while minimizing $H = \frac{1}{2} \Vert \ddot{q} + K_d \dot{q} \Vert ^2$. Projected Gradient (PG) with preferred acceleration $K_d \dot{q}$
	- singularities with minors method
	- augmented task with $\omega_{z,d}$
- ex 3: 2R planar constrained along a vertical segment between A and B
	- only one motor: underactuated robot, second link is passive
	- reduced dynamic
	- if the robot is in equilibrium in A what is the applied torque $\tau_0$
		- inverse kinematics of the 2R to find $q_A$
		- inverse constrained dynamic model to find $\tau$ given the reduced dynamic model
	- rest-to-rest from A to B with a sinusoidal acceleration, find $\tau_d$
		- since it's a reduce dynamic exercise, the motion is on $v$ and its derivatives. Precisely, we have $\dot{v} = \Delta sin(\omega t)$, integrating and adding constants $C$ in order to get a rest-to-rest velocity and a position $p_y$ starting from A and ending in B, we find the value of $\Delta$, precisely by imposing $p_y(T) = B$
		- after having found $v$ and $\dot{v}$ you can find $\tau_d$ with the inverse constrained dynamic model as before


## April 2022 (2022-04)
- ex 1: calibration of the two links of a 2R
	- you have the results of some expertiments: $q_i \to p_i$
	- compute $\hat{p} = p_{experiment} - J $\delta l$
	- now you have $\delta p$ and you can pseudoinvert, getting $\delta l = \Phi^{\verb|#|} \delta p
	- result is $l = \hat{l} + \delta l
	- there are all zeros and equal rows in the regressor matrix, due to singularities of $\Phi$. These rows can be eliminated
- ex 2: $\ddot{q}(t) = \ddot{q}_k$ for $t \in [t_k, t_k + T_c]$, thus $\dot{q}_{k+1} = \dot{q}_k + T_c \ddot{q}_k$
	- at time $t_k$, provide $\ddot{q}_k$ s.t. $\ddot{r}_{d,k}$ and minimize $H = \frac{1}{2} \Vert \dot{q}_{k+1} \Vert ^2 = \frac{1}{2} \Vert \dot{q}_{k} + T_c \ddot{q}_{k} \Vert ^2$. This is solved with Projected Gradient (PG) using preferred acceleration $\ddot{q}_k = - \frac{\dot{q}_{k}}{T_c}$, since it make the $H$ goes to zero. 
- ex 3: 3R spatial, moving frames algorithm:
	- compute M
	- note that $r_{ci, i}$ are defined with $L_i - dc_i$ since $dc_i starts from the position of the $i-1$-th joint
- ex 4: projected gradient (PG) on $\dot{q}$ with $H_{range}$ for keeping the joint ranges on the midranges. The solution is then $\dot{q} = \dot{q}_r + \dot{q}_n$
	- e.e. task scaling $\dot{q} = k\dot{q}_r + \dot{q}_n$ to find $k$
	- applying directly SNS on the entire $\dot{q}$ would not be correct since the solution $\dot{q}$ contains a null-space term that does not scale with $\dot{r}$ 
- ex 5: PR planar, Projected Gradient (PG) with two typical dynamic objectives $H$
	- with $H_A = \frac{1}{2} \Vert \tau \Vert ^2$, that is the torque norm
	- with $H_B = \frac{1}{2} \Vert \tau \Vert_{M^{-2}} ^2$, that is the squared inverse inertia weighted torque norm
- ex 6: PR planar 
	- gravity term $g(q)$
	- $\alpha$ upper bound of $\Vert \Nabla g(q) \Vert$



## June 2022 (2022-06)
- ex 1: PR planar
	- adaptive control law for smooth trajectory $q_d(t)$ having matrix $$ M(q) = 
\begin{bmatrix}
A & Bcos(q2) \\
Bcos(q2) & C \\
\end{bmatrix}
$$
		- use $A$, $B$ and $C$ as dynamic parameters
- ex 2: macro-micro planar 4R
	- two tasks, the one of the macro robot and the one of the micro robot. So you can create the extended jacobian $J_E$ and find its algorithmic singularities
	- Task Priority (TP)
- ex 3: robot with n elastic joints interacting with the environment, has a link dynamics and a motor dynamics
$$
\begin{cases}
M(q)\ddot{q} + S\dot{q} + g = \tau_J + J^{T}F \\
B\ddot{\theta} + \tau_J = \tau
\end{cases}
$$
	- design control law $\tau$ s.t. a desired motor dynamics holds: just extract $\ddot{\theta}$ from this dynamic and plug it in the motor equation
	- design link dynamics s.t. the desired impedance model holds: just plug the F from the desired impedance model in the classical cartesian dynamic model (first equation), getting the value of $u = \tau_J$ for which everything holds
	- then you can plug the value of $u$ in the $\tau$ contro law, getting the final control law that depends only on $q$, $\dot{q}$, $\theta$, $\dot{\theta}$
	- when an external constant force $F = \bar{F}$ is applied from the environment, find the expression of $x_E$, $u_E$ and $\tau_E$
- ex 4: one-link (pendulum) actuated and under gravity, with input torque bounds $\tau_{max}$
	- rest-to-rest motion from $\theta(0) = -\frac{pi}{2}$ and $\theta(T) = \frac{pi}{2}$ with bang-bang acceleration $\implies$ find minimum time T



## July 2022 (2022-07)
- ex 1: 3R spatial from $q$ to $p$ coordinates
	- find the transformation $J$ and compute $M_p$ and $c_p$
- ex 2: dynamic model partitioned in the part related to the first joint $q_1$, to whicfh is imposed an holonomic constraint $h(q) = q_1 - k$ and all the rest
	- reduced dynamics
	- two requests:
		- $\lambda = 0$: inverse formula to find $\tau_1$ s.t. $\lambda = 0$
		- regulation on $q_{2,d}$: inverse formula on the reduced dynamic (that depends only on $\tau_2$) in order to find $\tau_2$ and substitute a PD to $\ddot{q}_2$
- ex 3: PP planar
	- dynamic model
	- trajectory from $P_s$ to $P_g$ having bounds only on the forces $U_max$ $implies$ bang-bang trajectory on both joints since there are no bounds on the velocity $\dot{q}$. Indeed, the dynamic model of the robot has no dependence on $\dot{q}$, but only on $\ddot{q}$, so only bounds on $A_max$.
	- formulas to find $T*$ and $T_s$
	- **IMPORTANT NOTE**: the bounds are **asymmetric** since we have to make $U_max$ negative, not the entire expression $\ddot{q}$, so if we have another addend that doesn't depend on $U_max$ it will not be made negative and the bounds will be asymmetric, as in this case
	- is the cartesian path from $P_s$ to $P_g$ a linear path? No, indeed you can see that $\frac{\dot{y}}{\dot{x}} = \frac{\dot{q}_2}{\dot{q}_1} = k$ is a line until $T_{s,x}$, then you have two different curvilinear parts.



## September 2022 (2022-09)
- ex 1: 3R **with equal link** has to be in $p = 0$ minimizing joint range function $H(q)$
	- Projected Gradient
	- **TODO**: it seems that the only family of solutions are $(\ q_1 \ \frac{2pi}{3} \ \frac{2pi}{3}\ )$ and $(\ q_1 \ \ - frac{2pi}{3} \ \ - frac{2pi}{3}\ )$ so basically equilateral triangles orientated with $q_1$. But when I do the inverse kinematics I don't have these types of solution but (2pi/3, 2pi/3, 4.01something)
	- show that the robot, starting from $q(0)$ defined with the equilateral triangles above, converges to $\bar{q}$ s.t. $\nabla H(\bar{q}) \neq 0$ but $\dot{q} = 0$
- ex 2: RP planar robot
	- rest-to-rest cartesian trajectory from $P_i$ to $P_f$ with bang-coast-bang acceleration profile
	- **TODO**: it seems that using $u = M(\ddot{y} + PD) + g$ with $\ddot{y} =$ bang-bang trajectory is not the proper solution, why?
- ex 3: two masses and a pulley
	- case (a) we measure position only of the first mass ($\theta$) $\implies$ the only dynamic equation is $(M + B)\ddot{\theta} - M g_0$
	- case (b) we measure both position ($\theta$ and $q$) and we have a spring
	- **TODO**



## October 2022 (2022-10)
- RP robot
	- dynamic model
	- linear parametrization
	- regulation control law knowing only some parameters: you don't know the entire dynamic so you need to do the classic PD on $u(q)$ but you can find $\alpha$ s.t. there is global asymptotic stabilization with those known parameters
	- all parameters are known: so now you can use the dynamic. Don't use regulation since it give you "a given time T" so he want a rest-to-rest cubic in time T
	- **asymmetric** bang-bang: bang-bang since there is no constraint on $\dot{q}$


## January 2023 (2023-01)
- ex 1: 2R planar
	- dynamic model
	- PD control law: minimum values of $K_p$ and $K_d$
		- note that since there is viscous friction at both joints you can put $K_d = 0$
	- adaptive control law
	- bang-bang rest-to-rest motion: neglect gravity since we are in an horizontal plane now
- ex 2: 2P cartesian planar, hybrid force-velocity control law
	- dynamic model of then 2P
	- rotation from $RF0$ to $RF_{task}$, dividing $v$ and $f$ in tangential and normal
	- compliant environment with stiffness $K_n$ ($v_n$ with deformation $\delta_n$) and frictionless ($f_t = 0$)
	- control law **TODO**



## February 2023 (2023-02)
- ex 1: 3R planar torque controlled. These are just three cases of Linear Quadratic (LQ) optimization
	- $\tau_A$ that minimizes $H_A = frac{1}{2} \Vert \ddot{q} \Vert ^2$ is just the pseudoinverse considering $\ddot{q} = J^{\verb|#|}(\ddot{r} - \dot{J}\dot{q})$ = J^{\verb|#|}\ddot{r}$ at rest
	- $\tau_B$ that minimizes $H_A = frac{1}{2} \Vert \ddot{q} \Vert ^2$ but using absolute coordinates. Weighted matrix with W = $T^T T$
	- $\tau_C$ that minimizes $H_A = frac{1}{2} \ddot{q}^T M(q) \ddot{q}$ is inertia weighted matrix
- ex 2: single link under gravity (pendulum)
	- rest-to-rest swing-up maneuver with a cubic under torque bound $u_{max}$
	- find the minimum time $T*$
		- differently from ex 6 of 2023-04, here the gravity term depends on $\theta$, since $u = I\ddot{\theta} + mg_0dsin(\theta) = u_{inertia}(\theta) + u_{gravity}(\theta)$, while in 2023-04 we have isolated $\ddot{\theta}$ since $g$ was a constant
		- $u_{inertia}$ as in the other exercise, is linear and maximum in $t = 0$ and $t = T$, while $u_{gravity}$ is sinusoidal and maximum at the midpoint $T/2$, when $sin(\theta) = 1$. The superposition of the two torques will have a maximum in the first half of the motion, where both terms are positive
		- note also that the faster is the trajectory (i.e. the smaller T), the more $\ddot{\theta}$ will grow, and the more $u_{inertia}$ will dominate $u_{gravity}$. So when T is small enough, you can neglect $u_{gravity}$ and find $T$
	- minimum uniform time scaling factor, compute at time $t = 0$ since it's the time at which we have the overcome of the $u_max$
- ex 3: two masses and a spring in between
	- dynamic model
	- force control laws decentralized: $F_i$ depends only on $q_i$
	- find unique equilibrium $\bar{q}$ at steady state $(\ q \ \dot{q} \ ) = (\ \bar{q} \ 0 \ )$ for the closed-loop system under the control law
	- prove global asymptotic stbaility of the equilibrium state by Lyapunov/LaSalle
	- since $\bar{q}$ is not $q_d$, we modify the control law to enforce $\bar{q} = q_d$
		- straightforward: cancel the effect of elasticity adding $K e_p$ in the control laws. This would fully decouple the behavior of the two masses but we will not have the decentralization, since now $F_i$ depends also on $q_j$
		- to keep the decentralization we can use a PID, since it removes the constant steady-state error 
		- another solution is to use a feedforward term $F_{i, ffw}$


## April 2023 (2023-04)
- ex 1: SNS on acceleration with bounds both on velocity and acceleration
	- update of $\ddot{q}(t)$ done every $T_c$
	- $\ddot{Q}_min$ and $\ddot{Q}_max$ set
- ex 2: having DH of a 3R planar, find $r_{c,i}$ for each CoM s.t. $g(q)$ is a certain vector
	- since $g(q)$ has $g_1 = g_2 = 0$ we need $r_{cy,2} = r_{cy,3} = 0$
	- find general $g(q)$ for a 3R and find the relationship
- ex 3: 4P planar:
	- compute M
	- inertia-weighted pseudoinverse $J_M^{\verb|#|}$ to minimize T
	- pseudoinverse $J^{\verb|#|}$ to minimize $\Vert\dot{q}\Vert$
- ex 4: RPR spatial
	- compute M
- ex 5: given $M$ find $c$ and three different factorization of $S$ s.t. the first two creates a skew-symmetric matrix and the third doesn't
	- first S is the standard factorization, then the second is obtained adding a skew-symmetric matrix with the component of $\dot{q}$. The last, invalid one, is obtained changing a bit the added skew-symmetric matrix on the previous one
	- find the unique regressor matrix 
- ex 6: PR spatial
	- rest-to-rest cubic trajectory for both joints, find $T^*$ having bounds on torques




## June 2023 (2023-06)
- ex 1: 4R planar
	- inertia matrix with absolute coordinates and DH
	- from absolute to DH coordinates: $\theta_i = q_i - q_{i-1}$, or equivalently $q_i = theta_1 + … + theta_i$
	- $\tilde{M(\theta)} = T^T M(\theta) T$
	- angular velocity of link $i$ w.r.t absolute coordinates is just $\dot{q}_i$, while w.r.t DH coordinates is $\dot{theta}_1 + … + \dot{theta}_i$
	- nR robot generic expression of $T_i$ and so of $\Vert v_{ci} \Vert$
- ex 2: two tasks, Task Priority
	- execute just one, both simultaneously with and without priority
	- norm of the error is lower in the case without priority
- ex 3: PRR planar, PD + gravity compensation
	- $g(q)$: note that $q_3$ doesn't change $P_{c3}$
	- $K_{P,m} > \alpha$, actually since $g(q)$ first term is constant and third is zero we have that $K_{P,2} > \alpha$ while the others just need to be positive **CHECK WHY IS THAT**
- ex 4: 2R with payload and a vertical guide
	- $h(q) = p_x = 0$
	- Choose $D(q)$ s.t. with $A(q)$ forms the Jacobian $\implies$ determinant is simpler
	- $\tau$ control law using $\dot{v}_d$ found from quintic trajectory



## April 2024 (2024-04)
- ex 1: 3R planar with obstacle (clearance)
	- Projected Gradient PG: impose $v_e$ at .e.e while maximizing $H_{dist}$ that is distance from an obstacle
		- distance from the obstacle is norm between the closest robot point and closest circle point, then consider that you can rewrite it with $C$ and $r$
	- TP: two tasks, the first is the previous the second is velocity of the closest point $v_m$
		- case B: using $v_m$ equals to a velocity along the direction of the gradient of the clearance
- ex 2: 2R spatial
	- non-diagonal inertia since is not on the axis of joint **TO CHECK IF IT'S TRUE**
	- derive dynamic model
	- dynamic coefficient
	- find $\tau_d$ plugging $q_d(t)$ in $Y(q, \dot{q}, \ddot{q}) = \tau$
	- unforced ($\tau = 0$) equilibrium state at $(q_e \ 0)$ so at rest $\implies$ when $g(q) = 0$
	- mechanical parameters s.t. $g(q) = 0 \forall q$
	- force $F_e$ applied to the tip of the second link in the horizontal plane $(\ y_0 \ z_0 \ ), at rest and unforced $\implies M \ddot{q} = J_e^T F_e = \tau_e implies \ddot{q} = M^{-1} J_e^T F_e$ and so $\ddot{p}_e = J_e \ddot{q} = J_e M^{-1} J_e^T F_e$, where $ M_p = J_e M^{-1} J_e^T $ is the inverse of the inertia matrix at the e.e. level, but rescricted to the 2D horizontal plane of interest.
		- Note: if $M_p = J_e M^{-1} J_e^T $ were diagonal, $\ddot{p}$ would have been in the same direction of $F_e$ **TO VERIFY**
- ex 3: Newton-Euler for a 6R
	- free space vs subject to known active wrench
- ex 4: robot with torque limit $| \tau_i | \leq T_i \geq 5 * max_q | g_i |$ (i.e. the robot can sustain at least its own weight under gravity, with a conservative margin factor of 5). 
	- When $\dot{q} \neq 0$, we have that $\dot{E} = \dot{q}^T u(t_0)$. Since $| \tau_i | \leq T_i |$, the maximum instantaneous decrease of E is $\tau_{0,i} = - T_i * sign(q_i) \forall joint i$, if joint I is in motion, while the joint that are not moving we put $\tau = 0$.
	- When $\dot{q} = 0$, we induce a decrease with $\ddot{E} = \ddot{q}^T*\tau + \dot{q}^T \dot{\tau} = \ddot{q}^T \tau$ and since $M\ddot{q} + g = \tau$ we have that $\ddot{E} = \tau^T M^{-1}\tau - g^T * M^{-1} \tau. To find the maximum instantaneous decrease of E we compute $\nabla_{\tau} \ddot{E} = 0$, finding the corresponding $\tau_0$ 
	- You can extend this reasoning for higher order time derivatives of E: for instance, if we have $g(q) = 0$


## June 2024 (2024-06)
- ex 1: moving frames algorithm
	- from DH draw the robot
- ex 2: 4R planar Projected Gradient (PG)
	- task of pointing an object in the plane
	- **TO RECHECK**
- ex 3: 2n equations of n-joints robot and n motor equations
	- **TODO** 
- ex 4: PR planar
	- residuals
	- $\lim_{t \to \inf} r_1(t) = 2$



## July 2024 (2024-07)
- ex 1: PRR spatial moving frames algorithm	
	- put the last frame on CoM 3, not on the e.e. to simplify the $M(q)$
	- a 3-DoF robot has 30 parameters: 3 for each $dc_i$, 3 $m_i$ and 9-3 for each $I_i$ since is a symmetric matrix (so we just need diagonal and three elements
- ex 2: 2R planar
	- Projected Gradient but with $\ddot{q}_{PG}$, given $\ddot{r}_d(t)$ and cost H to minimize
	- Important note: since you want to find $\ddot{q}$ s.t. it minimizes H, namely the error between $\ddot{q}$ and $\dot{q}_0 = - K_v \dot{q}$, you can use directly $\dot{q}_0 =  - K_v \dot{q} in the calculation of $\ddot{q}_{PG}$ instead of $\nabla H$. Indeed, we don't need to follow the gradient of H since we have $\ddot{q}_0$ directly.
	- Note that as the damping is the derivative term, the $K_s$ spring is the $K_p$. So why are we using $K_p$ if the spring is present? Because we want a **specific** position $q_d$, while the spring is giving us something that is not exactly our position. The same holds for the derivative term: in the case we want to have specific desired error dynamics that do not match the one of the damping effect, we would have needed the $K_d$ to add
- ex 3: two masses, a pulley, damped elastic spring and viscous friction on the motion
	- dynamic model: with newton or lagrangian approach
	- simplest control law to regulate $q_d$
		- in steady state we have $\bar{\tau}$, $\bar{q}$ and $\bar{\theta}(q_d)$, where $\bar{\theta}$ was expressed in function of q since the problem requires a $q_d$. The condition on $\bar{\tau}$ and $\bar{\theta}$ must be satisfied by the controller
		- since there is are damping and viscous friction, you don't need the derivative term: the simplest feedback law is just $\tau = \tau_d + K_p(\theta_d - \theta)$, where \tau_d was find in the previous point $(\bar{\theta})$ and $\theta_d$ depends on $q_d$
		- verify asymptotic stability with Lyapunov and by local approximate linearization
		- inverse dynamic problem (find $\tau$ from $q$) having set D = 0
			- rewrite the dynamic model substituting $\theta$ with $q$. At the end you will have $\theta =$ something that depends only on $q$ and not on $\theta$ anymore.

- ex 4: Pendubot (underactuated 2R, so just one joint providing torque $\tau$)
	- find all forced equilibria $(\bar{q}, 0)$, basically $g(q) = 0$
		- we found $\bar{q}$ and corresponding $\bar{\tau}$ needed
		- PD with gravity compensation is hard since the robot is underactuated: we locally asymptotically stabilizes around $(\bar{q}, 0)$ with linearization of the dynamic model
			- Controllability Matrix
	