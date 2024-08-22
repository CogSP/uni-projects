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

## June 2016 - Final Test with Midterm (2016-07-midterm)
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


## October 2016 (2016-10)
- ex 1: RP planar:
	- dynamic model with uniform mass distribution
	- equilibrium configuration under $u = 0$, i.e. $q$ s.t. $g(q) = 0$

## March 2017 (2017-03)
- ex 1: 3R spatial robot: determine $M(q)$ using moving frames
	- minimal number of dynamic coefficients


## September 2017 (2017-09)
- ex 1: RP planar
	- dynamic model: $M$, $c$ and $g$
	- plane $(x_0, y_0)$ inclined by $\alpha$ w.r.t. horizontal plane around $x_0$
	- $\ddot{q}_0$ s.t. robot at rest, i.e. $\dot{q}_0 = 0$ and $u = (\tau, F)$
	- $\Vert\ddot{p_0}\Vert^2$ and $q_{min}^{\*}$ and $q_{max}^{\*}$, given bounds on $\tau$, $F$ and $q_2$ 	
		- how $min\Vert\ddot{p}_0\Vert^2$ and $max\Vert\ddot{p}_0\Vert^2$ change if $\frac{I_1}{I_2}$ changes

## April 2018 (2018-04)
- ex 1: planar PPRR
	- dynamic model with viscous friction
	- linear parametrization $Y(q, \dot{q}, \ddot{q})a = u$

## April 2019 (2019-04)
- ex 2: PRP planar
	- dynamic model
	- linear parametrization $Y(q, \dot{q}, \ddot{q})a = u$

## June 2019 (2019-06)
- ex 1: 6R spatial (Kawasaki S030)
	- find $g(q)$: use DH or guess that $g_1 = g_4 = g_5 = g_6 = 0$
	- linear parametrization
	- $q_e$ s.t. $g(q_e) = 0$

## April 2020 (2020-04)
- ex 7: PPR planar
	- dynamic model
	- linear parametrization

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
