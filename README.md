# RL-methods-for-CSTR

We tested different Reinforcement Learning models, especially those with an actor-critic structure, on a CSTR example to maintain the closed-loop state within the stability region.

## 1. Continuous Stireed Tank Reactor (CSTR) Example

- Let us consider a first-order, exothermic, irreversible reaction from A to B






- The First Principle equation for this system are as follows:
- $$\frac{dC_A}{dt} = \frac{F}{V_L} (C_{A0} - C_A) - k_0 e^{-\frac{E}{RT}} C_A$$
- $$\frac{dT}{dt} = \frac{F}{V_L} (T_0 - T) + \frac{-\Delta H}{\rho_L C_P} k_0 e^{-\frac{E}{RT}} C_A + \frac{Q}{\rho_L C_P V_L} $$

  

     
     <img src="https://github.com/GuoQWu/RNN-based-MPC/assets/85721266/ccfdf6cd-f984-4232-8dd4-1b1c4e5e84e4" width="300" height="300">



- Where,

   - 𝐶_𝐴: Concentration of reactant A (kmol/m3)
   - 𝑇: Temperature of the reactor (K)
   - 𝐶_𝐴0: Concentration of reactant A in the feed
   - 𝑄 :  Heat input rate (kJ/h)
   - F: feed volumetric flow rate (m3/h)
   - 𝑇0: Inlet temperature (K)


- The State and Manipulated variable for this system is:

    - States variables: 𝐱=[𝐶_𝐴−𝐶_𝐴𝑠, 𝑇 −𝑇_𝑠]
    - Control / Manipulated variables: 𝐮=[𝐶_𝐴0−𝐶_𝐴0𝑠, 𝑄 −𝑄_𝑠]


## 2. Actor-Critic Structure

- Actor-critic methods are a class of algorithms in reinforcement learning (RL) that combine two neural networks: the actor and the critic. These methods aim to address the limitations of pure policy-based or value-based methods by integrating the strengths of both.

- The model input and output are as follows:
    - Input: System initial state variable 𝐱, and control variables 𝐮.
    - Output: An estimation of current value and a proper control at state input 𝐱.


## 3. Current Methods

- Relaxed Continuous-Time Actor-critic (RCTAC): it's able to lead an arbitrary initial policy to a nearly optimal policy, even for general nonlinear input non-affine system dynamics

- Deep Deterministic Policy Gradient (DDPG): it requires only a straightforward actor-critic architecture and learning algorithm with very few “moving parts”, making it easy to implement and scale to more difficult problems and larger networks.


## 4. Citations

CSTR example:
Wu, Z., & Christofides, P. D. (2019). Handling bounded and unbounded unsafe sets in control Lyapunov-barrier function-based model predictive control of nonlinear processes. Chemical Engineering Research and Design, 143, 140-149.

RCTAC alorithm:
Duan, J., Li, J., Ge, Q., Li, S. E., Bujarbaruah, M., Ma, F., & Zhang, D. (2023). Relaxed actor-critic with convergence guarantees for continuous-time optimal control of nonlinear systems. IEEE Transactions on Intelligent Vehicles, 8(5), 3299-3311.

DDPG algorithm:
Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.
https://github.com/vikash9899/Contorl-CSTR-using-Reinforcement-learning/tree/main
