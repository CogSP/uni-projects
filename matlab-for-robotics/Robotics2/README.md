# matlab-for-robotics - Robotics 2

## Misc
- dynamic_model_planar: inputting T and the number of joints, compute
	- $M(q)$
	- $c(q, \dot{q})$
	- $g(q)$

## June 2010 (2010-06)
- RP Robot: dynamic model, redundancy (one-dimensional task) -> pseudoinverse
	- weight matrix $W$ to compare $\tau$ revolute joint torque and prismatic joint force
	- null space method to not build up joint velocities (TODO: check if this is actually the null space method)

## September 2010 (2010-09)
- RPR Robot: find inertia matrix and minimal set of dynamic coefficients


## April 2016 (2016-04)
- ex 1: PRR Robot: dynamic model 
- ex 2: 4R planar
	- find $g(q)$
	- equilibrium configuration $q_e$ s.t. $g(q_e) = 0$
	- linear parametrization $g(q) = Y_G(q)a_G$
	- $d_i$ s.t. $g(q) = 0 \forall q$, i.e. $a_G = 0$
- ex 3: 4R planar, determine $\dot{q}$ s.t. $\dot{r}_d = (v_d, \dot{\phi}_d)$ is realized while decreasing $H_{range}(q)$


## March 2017 (2017-03)
- ex 1: 3R spatial robot: determine $M(q)$ using moving frames
	- minimal number of dynamic coefficients


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