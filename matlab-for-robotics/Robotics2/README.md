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

## September 2010 (2010-09)
- RPR Robot: find inertia matrix and minimal set of dynamic coefficients


## September 2010 (2010-09)
- ex 1: RPR planar:
	- find $M$
	- minimal set of dynamic coefficient $a \in \mathcal{R}^p$ for $M$


## April 2016 (2016-04)
- ex 1: PRR Robot: 
	- dynamic model
- ex 2: 4R planar
	- find $g(q)$
	- equilibrium configuration $q_e$ s.t. $g(q_e) = 0$
	- linear parametrization $g(q) = Y_G(q)a_G$
	- $d_i$ s.t. $g(q) = 0 \forall q$, i.e. $a_G = 0$
- ex 3: 4R planar, determine $\dot{q}$ s.t. $\dot{r}_d = (v_d, \dot{\phi}_d)$ is realized while decreasing $H_{range}(q)$

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

## April 2020 (2020-04)
- ex 7: PPR planar
	- dynamic model
	- linar parametrization

## April 2021 (2020-04)
- ex 1: 2R planar
	- dynamic model
	- linear parametrization

## April 2022 (2022-04)
- ex 3: 3R spatial:
	-  compute M

# April 2023 (2023-04)
- ex 3: 4P planar:
	- compute M
	- inertia-weighted pseudoinverse $J_M^{\verb|#|}$ to minimize T
	- pseudoinverse $J^{\verb|#|}$ to minimize $\Vert\dot{q}\Vert$
- ex 4: RPR spatial
	- compute M
