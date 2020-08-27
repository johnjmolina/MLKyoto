# last
import jax.numpy as np
from functools import partial
from jax import jit
from  . import phi


class _etdrk:
    def __init__(self, *, G, dt):
        self.G  = G
        self.dt = dt

class etdrk1(_etdrk):
    def __init__(self, *, L, G, dt, dps=100):
        super().__init__(G=G, dt=dt)
        hL = dt*L.reshape(-1)
        self.phi = np.array([phi.phin(n, hL, dps=dps).reshape(L.shape) for n in range(2)])

    @partial(jit, static_argnums=(0,))
    def step(self, u):
        u = self.phi[0] * u + self.dt * self.phi[1] * self.G(u)
        return u


class etdrk2(_etdrk):
    def __init__(self, *, L, G, dt, dps=100):
        super().__init__(G=G, dt=dt)
        hL   = dt*L.reshape(-1)
        hLh  = hL / 2
        phi1 = phi.phin(1, hL, dps=dps).reshape(L.shape)
        phi2 = phi.phin(2, hL, dps=dps).reshape(L.shape)
        
        self.phi  = np.array([phi.phin(n, hL,  dps=dps).reshape(L.shape) for n in range(1)])
        self.phih = np.array([phi.phin(n, hLh, dps=dps).reshape(L.shape) for n in range(2)])
        self.hc1  = self.dt*(phi1-2.0*phi2)
        self.hc2  = 2.0*self.dt*phi2

    @partial(jit, static_argnums=(0,))
    def step(self, u):
        # G1
        G1 = self.G(u)

        # U2, G2
        Ui = self.phih[0]*u + self.dt*(0.5*self.phih[1]*G1)
        G2 = self.G(Ui)

        # u_{n+1}
        u = self.phi[0]*u + self.hc1*G1 + self.hc2*G2
        return u

class etdrk3(_etdrk):
    """ Heun's method : worst case order 2.75"""
    def __init__(self, *, L, G, dt, dps=100):
        super().__init__(G=G, dt=dt)
        hL         = dt*L.reshape(-1)
        hL13       = hL/3.0
        hL23       = 2.0*hL13
        phi1       = phi.phin(1, hL,   dps=dps).reshape(L.shape)
        phi1_13    = phi.phin(1, hL13, dps=dps).reshape(L.shape)
        phi1_23    = phi.phin(1, hL23, dps=dps).reshape(L.shape)
        phi2       = phi.phin(2, hL,   dps=dps).reshape(L.shape)
        phi2_23    = phi.phin(2, hL23, dps=dps).reshape(L.shape)

        self.phi   = np.array([phi.phin(n, hL,  dps=dps).reshape(L.shape) for n in range(1)])
        self.phi13 = np.array([phi.phin(n, hL13,dps=dps).reshape(L.shape) for n in range(1)])
        self.phi23 = np.array([phi.phin(n, hL23,dps=dps).reshape(L.shape) for n in range(1)])
        self.hc1   = self.dt*(phi1 - 1.5*phi2)
        self.hc3   = self.dt*1.5*phi2
        self.hc1_2 = self.dt/3.0*phi1_13
        self.hc1_3 = self.dt/3.0*(2.0*phi1_23 - 4.0*phi2_23)
        self.hc2_3 = self.dt/3.0*(4.0*phi2_23)

    @partial(jit, static_argnums=(0,))
    def step(self, u):
        #G1
        G1 = self.G(u)

        #U2(G1), G2
        Ui = self.phi13[0]*u + self.hc1_2*G1
        G2 = self.G(Ui)

        #U3(G1, G2), G3
        Ui = self.phi23[0]*u + self.hc1_3*G1 + self.hc2_3*G2
        G3 = self.G(Ui)

        #u_{n+1}(G1, G3)
        u = self.phi[0]*u + self.hc1*G1 + self.hc3*G3
        return u
        

class etdrk45(_etdrk):
    """ Hochbruck and Ostermann's fourth order ETDRK method"""
    def __init__(self, *, L, G, dt, dps=100):
        super().__init__(G=G, dt=dt)
        # temporary data
        hL    = dt*L.reshape(-1)
        hLh   = hL/2
        phi3  = phi.phin(3, hL, dps=dps).reshape(L.shape)
        phih3 = phi.phin(3, hLh,dps=dps).reshape(L.shape)
        
        # persistent data
        self.phi  = np.array([phi.phin(n, hL,  dps=dps).reshape(L.shape) for n in range(3)])
        self.phih = np.array([phi.phin(n, hLh, dps=dps).reshape(L.shape) for n in range(3)])
        self.a52  = phi.phi_a52(phi2_dt = self.phi[2], phi3_dt = phi3, phi2_hdt = self.phih[2], phi3_hdt = phih3)
        self.hc1  = self.dt*(self.phi[1] - 3*self.phi[2] + 4*phi3)
        self.hc4  = self.dt*(4*phi3 - self.phi[2])
        self.hc5  = self.dt*(4.0*self.phi[2] - 8.0*phi3)

    @partial(jit, static_argnums=(0,))        
    def step(self, u):
        uh = self.phih[0]*u

        #G1
        G1 = self.G(u)

        #U2(G1),G2 
        Ui = uh + self.dt*(0.5*self.phih[1]*G1)
        G2 = self.G(Ui)

        #U3(G1, G2),G3
        Ui = uh + self.dt*((0.5*self.phih[1]-self.phih[2])*G1 + self.phih[2]*G2)
        G3 = self.G(Ui)

        #U4(G1, G2, G3),G4
        Ui = self.phi[0]*u + self.dt*((self.phi[1] - 2*self.phi[2])*G1 + self.phi[2]*(G2 + G3))
        G4 = self.G(Ui)

        #U5(G1, G2, G3, G4),G5
        Ui = uh + self.dt*((0.5*self.phih[1] - 0.25*self.phih[2] - self.a52)*G1 + self.a52*(G2 + G3) +  (0.25*self.phih[2] - self.a52)*G4)
        G5 = self.G(Ui)

        # u_{n+1}(G1, G4, G5)
        u  = self.phi[0]*u + self.hc1*G1 + self.hc4*G4 + self.hc5*G5
        return u
