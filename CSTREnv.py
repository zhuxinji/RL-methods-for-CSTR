import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CSTREnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, is_train=True):
        super().__init__()

        self.is_train = is_train
        self.dt = 1e-2
        self.EPISODE_LENGTH = 100

        # ---------- Action space ----------
        self.action_space = spaces.Box(
            low=np.array([-3.5, -5*1e5], dtype=np.float32),
            high=np.array([3.5, 5*1e5], dtype=np.float32),
            dtype=np.float32
        )

        # ---------- Observation space (physical bounds) ----------
        self.observation_space = spaces.Box(
            low=np.array([-2, -100.0], dtype=np.float32),
            high=np.array([2, 100.0], dtype=np.float32),
            dtype=np.float32
        )

        # ---------- Lyapunov matrix ----------
        self.P = np.array([[1060, 22],
                           [22, 0.52]])

        # ---------- Generate admissible initial states ----------
        self.initial_points = self._generate_initial_points()

        # ---------- Steady state ----------
        self.setpoint_states = np.array([0.0, 0.0])
        self.setpoint_actions = np.array([0.0, 0.0])

        self.episode = -1

        self.reset()

    # ==========================================================
    # Initial state sampling
    # ==========================================================
    def _generate_initial_points(self):
        fir = np.linspace(-2, 2, 40)
        sec = np.linspace(-100.0, 100.0, 200)
        pts = []

        for i in fir:
            for j in sec:
                x = np.array([i, j])
                if x @ self.P @ x < 372:
                    pts.append(x)

        #random.shuffle(pts)
        return pts

    # ==========================================================
    # Dynamics (Model 1)
    # ==========================================================

    def _dx_model(self, x, u):
        T0=300; V=1; F=5; E=5*(10**4); k0=8.46*(10**6); deltaH=-1.15*(10**4); Cp=0.231; rhoL=1000; R=8.314; CA0s=4; Qs=0;
        CAs=1.95; Ts=402;
        x1, x2 = x
        u1, u2 = u
        dx1 = (F/V)*(u1+CA0s-x1-CAs) - k0*np.exp(-E/(R*(x2+Ts)))*(x1+CAs)**2
        dx2 = (F/V)*(T0-Ts-x2) + (-deltaH/(rhoL*Cp))*k0*np.exp(-E/(R*(x2+Ts)))*(x1+CAs)**2 + (u2+Qs)/(rhoL*Cp*V)
        dx = np.array([dx1, dx2])
        return dx
    
    # ==========================================================
    # Step
    # ==========================================================
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        dx = self._dx_model(self.state, action)
        next_state = self.state + self.dt * dx

        # ---------- Reward ----------
        self.Q = np.array([[1, 0],[0, 0.01]])
        self.R = np.array([[0, 0],[0, 0]])
        
        reward = -(next_state@self.Q@next_state + action@self.R@action)

        # ---------- Termination ----------
        self.ep_step += 1
        
        steady = next_state@self.P@next_state < 2
        self.steady_history.append(steady)
        terminated = False
        truncated = self.ep_step >= self.EPISODE_LENGTH
        self.state = next_state

        if self.is_train:
            if len(self.steady_history) > 20:
                self.steady_history.pop(0)
                terminated = all(self.steady_history)
            if terminated:
                print()
                print(f'Steady State Reached! state:{self.state}, u:{action}, reward:{reward}')
                print()
            elif truncated:
                print(f'Episode:{self.episode}. state:{next_state}, u:{action}, reward:{reward}')
        
        return next_state, reward, terminated, truncated, {}

    # ==========================================================
    # Reset
    # ==========================================================
    def reset(self, *, seed=None, options=None, state=np.array([-0.6, 35])):
        super().reset(seed=seed)

        self.episode += 1
        self.ep_step = 0
        self.steady_history = []

        self.state = state
        #self.state = self.np_random.choice(self.initial_points).copy()

        return self.state, {}
