# matlab-for-robotics - Robotics 2

## Misc
- dynamic_model_planar: inputting T and the number of joints, compute
	- $M(q)$
	- $c(q, \dot{q})$
	- $g(q)$
- moving_frames_algorithm: find the dynamic model for spatial robot 

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
- **TODO** stuff about collision


## April 2016 (2016-04)
- ex 1: PRR Robot: 
	- dynamic model
- ex 2: 4R planar
	- find $g(q)$
	- equilibrium configuration $q_e$ s.t. $g(q_e) = 0$
	- linear parametrization $g(q) = Y_G(q)a_G$
	- $d_i$ s.t. $g(q) = 0 \forall q$, i.e. $a_G = 0$
- ex 3: 4R planar
	- PG to execute $r(t)$ while increasing $-H_{range}(q)$


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
- ex 4: reduced dybamic model after partition $q = (\ q_a \ q_b \ )$


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
	- torque limit not satisfied: time scaling: inertial torque scales with $k^2$
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
- ex 6: TO COMPLETE TODAY, 3/09


## April 2021 (2020-04)
- ex 1: 2R planar
	- dynamic model
	- linear parametrization

## April 2022 (2022-04)
- ex 3: 3R spatial:
	-  compute M


## April 2023 (2023-04)
- ex 3: 4P planar:
	- compute M
	- inertia-weighted pseudoinverse $J_M^{\verb|#|}$ to minimize T
	- pseudoinverse $J^{\verb|#|}$ to minimize $\Vert\dot{q}\Vert$
- ex 4: RPR spatial
	- compute M

## June 2023 (2023-06)
- ex 1: 4R planar
	- inertia matrix with absolute coordinates and DH
	- from absolute to DH coordinates: $\theta_i = q_i - q_{i-1}$, or equivalently $q_i = theta_1 + … + theta_i$
	- $\tilde{M(\theta)} = T^T M(\theta) T$
	- angular velocity of link $i$ w.r.t absolute coordinates is just $\dot{q}_i$, while w.r.t DH coordinates is $\dot{theta}_1 + … + \dot{theta}_i$
	- nR robot generic expression of $T_i$ and so of $\Vert v_{ci} \Vert$
- ex 2: two tasks
	- execute just one, both simultaneously with and without priority
	- norm of the error is lower in the case without priority
- ex 3: PRR planar, PD + gravity compensation
	- $g(q)$: note that $q_3$ doesn't change $P_{c3}$
	- $K_{P,m} > \alpha$, actually since $g(q)$ first term is constant and third is zero we have that $K_{P,2} > \alpha$ while the others just need to be positive **CHECK WHY IS THAT**
- ex 4: 2R with payload and a vertical guide
	- $h(q) = p_x = 0$
	- Choose $D(q)$ s.t. with $A(q)$ forms the Jacobian $\implies$ determinant is simpler
	- $\tau$ control law using $\dot{v}_d$ found from quintic trajectory