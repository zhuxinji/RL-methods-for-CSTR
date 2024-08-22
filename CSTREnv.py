import numpy as np
import torch
import math
import random
import gym
from gym import spaces
episode_length = 3000

class cstr_env(gym.Env):

    def __init__(self, order):  
        self.action_space = spaces.Box(low = np.array([-1., -0.0167], dtype=np.float32), 
                                       high = np.array([1., 0.0167], dtype=np.float32), 
                                       dtype=np.float32, shape=(2, ))   
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), 
                                            high=np.array([1.0, 1.0], dtype=np.float32), 
                                            dtype=np.float32, shape=(2, ))
        self.n_episode = 0 # current episode number.
        self.order = order
        P = torch.tensor([[9.35, 0.41], [0.41, 0.02]])
        torch.manual_seed(0)
        fir = np.linspace(-0.5, 0.5, 100)
        sec = np.linspace(-10.0, 10.0, 400)
        x = []
        for i in fir:
            for j in sec:
                t = torch.tensor([i, j]).to(torch.float32)
                if torch.sum(t@P*t)<0.3:
                    x.append([i, j])
        random.shuffle(x)
        self.initial_points = x

    def is_done(self, x_next):
        done = False
        c1 = (abs(x_next[0] - self.setpoint_states[0]) < 0.01)
        c2 = (abs(x_next[1] - self.setpoint_states[1]) < 0.01)
        steady_state = c1 and c2  

        # Record the steady state status for the current step
        self.goal_state_done[self.ep_step] = steady_state
        
        if self.ep_step > 5: 
            p5 = self.goal_state_done[self.ep_step-4]
            p4 = self.goal_state_done[self.ep_step-3]
            p3 = self.goal_state_done[self.ep_step-2] 
            p2 = self.goal_state_done[self.ep_step-1] 
            p1 = self.goal_state_done[self.ep_step-0] 
            # If the last three steps were steady states, set 'done' to True
            if  p5 and p4 and p3 and p2 and p1:
                done = True  
        return done 

    def sontag(self, state):
        x = torch.tensor(state, requires_grad=True).to(torch.float32).unsqueeze(0)
        x1, x2 = state[0], state[1]
        params = [0.5734, 395.3268, 100e-3, 0.1, 72e+9, 8.314e+4, 8.314, 310, -4.78e+4, 0.239, 1000, 1]
        #params = [0.9992, 310.1428, 100e-3, 0.1, 72e+9, 8.314e+4, 8.314, 310, -4.78e+4, 0.239, 1000, 1]
        CAs, Ts, CF, CV, Ck0, CE, CR, CT0, CDh, Ccp, Crho, CA0s = params
        f1 = (CF / CV) * (- x1) - Ck0 * np.exp(-CE / (CR * (x2 + Ts))) * (x1 + CAs)+(CF / CV) * (CA0s-CAs)
        f2 = (CF / CV) * (-x2) + (-CDh / (Crho * Ccp)) * Ck0 * np.exp(-CE / (CR * (x2 + Ts))) * (x1 + CAs) + CF*(CT0-Ts)/CV
        f = torch.tensor([[f1], [f2]]).to(torch.float32)
        g1 = CF / CV
        g2 = 1 / (Crho * Ccp * CV)
        
        V0 = 9.35*x1*x1+0.82*x1*x2+0.02*x2*x2
        dVdX = torch.tensor([[2*9.35*x1+0.82*x2, 0.82*x1+0.02*2*x2]]).to(torch.float32)
        LfV = dVdX@f
        
        LgV1 = g1 * dVdX[0][0]
        LgV2 = g2 * dVdX[0][1]
        
        if (abs(x1) < 0.1) and (abs(x2) < 1):
            deta_u = 0.001*(abs(x1)+abs(x2))
        else:
            deta_u = 0.0011
        Q = torch.tensor([[1., 1.], [1., 1.]])
        xx = torch.tensor([[x1, x2]]).to(torch.float32)
        xxt = xx.t()
        QX = xx  @ Q @ xxt
    
        invR1, invR2 = 1, 1
        kx1 = (LfV+math.sqrt(math.pow(LfV, 2) + invR1*math.pow(LgV1, 2)*(QX))) / (math.pow(LgV1, 2)+deta_u)
        ub1 = -kx1*LgV1
        kx2 = (LfV+math.sqrt(math.pow(LfV, 2) + invR2*math.pow(LgV2, 2)*(QX))) / (math.pow(LgV2, 2)+deta_u)
        ub2 = -kx2*LgV2
        #-- control input---#
        
        if (abs(LgV1) > 1e-5):
            ub1 = ub1[0][0]
        else:
            ub1 = 0
        
        if (abs(LgV2) > 1e-5):
            ub2 = ub2[0][0]
        else:
            ub2 = 0
        
        ua_ = torch.tensor([[ub1], [ub2]])
        u1b = 1
        u2b = 0.0167
        ua_[0][0] = np.sign(ua_[0][0])*min(abs(ua_[0][0]), u1b)
        ua_[1][0] = np.sign(ua_[1][0])*min(abs(ua_[1][0]), u2b)
        return np.array([ua_[0][0], ua_[1][0]])
    
    def get_dx1(self, x, u):
        params = [0.5734, 395.3268, 100e-3, 0.1, 72e+9, 8.314e+4, 8.314, 310, -4.78e+4, 0.239, 1000, 1]
        #params = [0.9992, 310.1428, 100e-3, 0.1, 72e+9, 8.314e+4, 8.314, 310, -4.78e+4, 0.239, 1000, 1]
        CAs, Ts, CF, CV, Ck0, CE, CR, CT0, CDh, Ccp, Crho, CA0s = params                   
        g1, g2 = CF/CV, 1/(Crho*Ccp*CV)
        x1, x2 = x[0], x[1]
        
        f1 = (CF/CV)*(-x1) - Ck0*np.exp(-CE/(CR*(x2+Ts))) * (x1+CAs)+(CF/CV) * (CA0s-CAs)
        f2 = (CF/CV)*(-x2) + (-CDh/(Crho*Ccp))*Ck0*np.exp(-CE/(CR*(x2+Ts)))*(x1+CAs) + CF*(CT0-Ts)/CV
        dx = [f1, f2] + u*[g1, g2]
        return dx

    def get_dx2(self, x, u):
        CT0=300; CV=1; CF=5; CE=5*(10**4); Ck0=8.46*(10**6); CdetaH=-1.15*(10**4); CCp=0.231; CrolL=1000; CR=8.314; CA0s=4; Qs=0;
        CAs= 3.9364; Ts=303.1661;
        Cp1=(CF/CV); Cp2=Ck0; Cp3=Cp1*(CA0s-CAs); Cp4=(-CdetaH/(CrolL*CCp)); Cp5=Cp1*(CT0-Ts); Cp6= (1/(CrolL*CCp*CV));
        x1, x2 = x[0], x[1]
        f1 = -Cp1*x1 - Cp2*np.exp(-CE/(CR*(x2+Ts)))*((x1+CAs)**2) + Cp3
        f2 = -Cp1*x2 + Cp4*Cp2*np.exp(-CE/(CR*(x2+Ts)))*((x1+CAs)**2) + Cp5 + Qs*Cp6
        g1 = Cp1
        g2 = Cp6
        dx = [f1, f2] + u*[g1, g2]
        return dx

    def normalize_minmax_states(X:np.ndarray):
        act_range_min = np.array([0.3, 10], dtype=float)   
        act_range_max = np.array([0.3, 10], dtype=float)  
        transform_range_min = np.array([-1., -1], dtype=float)  
        transform_range_max = np.array([1., 1.], dtype=float)   
    
        X_std = (X-act_range_min) / (act_range_max - act_range_min) 
        X_scaled = X_std * (transform_range_max - transform_range_min) +  transform_range_min 
        return X_scaled 
    
    def step(self, action):
        dt = 5e-3
        action = np.array(action)
        self.current_u = action
        state = self.current_s
        if self.order == 1:
            dx = self.get_dx1(self.current_s, action)
        else:
            dx = self.get_dx2(self.current_s, action)
        x_next = self.current_s + dt*dx
        done = self.is_done(x_next)
        #reward = - np.sum((x_next - self.setpoint_states)**2)* - np.sum((self.current_u - self.setpoint_actions)**2)  
        reward = -x_next[0]**2 - x_next[1]**2*0.001# - np.sum((self.current_u - self.setpoint_actions)**2)
        
        self.previous_u = self.current_u 
        self.current_s = x_next 
        self.ep_step += 1  

        # this is the trancated condition. 
        truncated = False 
        if self.ep_step == episode_length:
            truncated = True

        if self.ep_step == episode_length-1 or done:      
            self.n_episode += 1 
        
        # if done is true i.e. terminated is equal to done. 
        terminated = done

        return x_next, reward, done

    def reset(self, e):
        self.ep_step = 0 
        self.current_u= None 
        self.previous_u = None 
        self.current_s = None 

        ## list of true false which stores the weather the state is near to the goal state or not. 
        self.goal_state_done = [False] * (episode_length+5)

        self.setpoint_states  =  np.array([.0, .0], dtype=float)     
        self.setpoint_actions =  np.array([.0, .0], dtype=float)

        # this is the fixed initial state. 
        state, action = np.array([0.2, -5]),  np.array([0., 0.])
        self.current_u = action 
        self.previous_u = action  
        self.current_s = state  

        return state
