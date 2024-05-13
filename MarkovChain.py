import numpy as np
from DiscreteD import DiscreteD
import scipy.stats
from GaussD import GaussD
from HMM import HMM
class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob, emission_prob = None):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]
        self.B = emission_prob

        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        """if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True"""
        #i is transition beginning state
        for i in range(len(transition_prob)):
            #j is transition end state
            for j in range(len(transition_prob[0])):
                if transition_prob[i][j] == 1:
                     self.is_finite = True
                     self.final_state = i+1


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        S = []
        initialDistr = DiscreteD(self.q)
        currentState = initialDistr.rand(1)[0]
        
        for _ in range(tmax):
            S.append(currentState)  #Append current state to the sequence
            
            if self.is_finite and currentState == self.nStates + 1:
                #If finite-duration and END state is reached, stop early
                break
            
            #Transition to the next state based on the current state
            transitionDistr = DiscreteD(self.A[currentState - 1])  #Adjust index for 0-based
            currentState = transitionDistr.rand(1)[0]  #Generate next state
            
            if self.is_finite and currentState == self.nStates + 1:
                #If the next state is the END state, don't include it in the output
                S.pop()
                break
            
        return np.array(S)  #Convert to numpy array for consistency with class description

        

    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass
    def forward_logprob(self, pX):
        if self.B == None:
            pX = pX
        else:
            pX = self.B
        nStates = self.nStates
        nSamples = pX.shape[1]
        logAlpha = np.full((nStates, nSamples), -np.inf)  # Initialize with log of zero
        c = np.zeros(nSamples)  # Scaling factors to prevent underflow

        # Initialize first column of logAlpha
        logAlpha[:, 0] = np.log(self.q) + np.log(pX[:, 0])

        # Compute forward log probabilities
        for t in range(1, nSamples):
            for j in range(nStates):
                logAlpha[j, t] = np.log(pX[j, t]) + np.logaddexp.reduce(logAlpha[:, t-1] + np.log(self.A[:, j]))

        # The log likelihood of the observed sequence
        logLikelihood = np.logaddexp.reduce(logAlpha[:, -1])
        return logLikelihood
    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass
    #Min
    def forward(self, pX, is_finite = False):
        epsilon = 1e-12
        #alpha_j,t = P(X1 = x1, ....X_t = x_t, S_t = j| lambda)
        #alpha_hat_j,t = P(S_t = j, X_t = x_t |X1 = x1, ....X_t = x_t, lambda)
        if self.B == None:
            pX = pX
        else:
            pX = self.B
        nStates = self.nStates
        nSamples = pX.shape[1]
        alpha_hat = np.zeros((nStates, nSamples))
        alpha_temp = np.zeros(nStates)
        c = np.zeros(nSamples + 1)
        #pX is emission matrix with pX[j ,t] being P(X = x | S = j)
        alpha_temp = self.q * pX[:, 0]
        c[0] = np.sum(alpha_temp)
        alpha_hat[:, 0] = alpha_temp / c[0]

    
      

        for t in range(1, nSamples):
            alpha_temp = np.zeros(nStates)  # Re-initialize the temporary variable
        
            for j in range(nStates):
                # Compute alpha_temp for each state at the current time step
                alpha_temp[j] = pX[j, t] * np.sum(alpha_hat[:, t - 1] * self.A[:, j])

            # Calculate the scale factor for this time step
            c[t] = np.sum(alpha_temp)
        
            # Scale alpha_hat for the current time step
            alpha_hat[:, t] = alpha_temp / c[t]
        if is_finite == True:
            sum_final_alpha = np.sum(alpha_hat[:, -1] * self.A[:, nStates])  # Apply transition to the final state
            c[-1] = sum_final_alpha
        
        return alpha_hat, c


    def finiteDuration(self):
        pass
    
    def backward(self, c, pX, is_finite = False):
        if self.B == None:
            pX = pX
        else:
            pX = self.B
        epsilon = 1e-10
        nStates = self.nStates
        nSamples = pX.shape[1]

        # Initialize beta_hat
        beta_hat = np.zeros((nStates, nSamples))

        if is_finite:
            beta_hat[:, -1] = self.A[:, -1]  #Initialize with transition probabilities to an absorbing state
            #Normalize beta_hat for the last time step
            beta_hat[:, -1] /= (c[-1] * c[-2])
        else:
            beta_hat[:, -1] = 1.0 / (c[-1] + epsilon)

        #Backward recursion
        for t in range(nSamples - 2, -1, -1):
            for i in range(nStates):
                sum_products = 0
                for j in range(nStates):
                    sum_products += self.A[i, j] * pX[j, t + 1] * beta_hat[j, t + 1]
                
                #Update beta_hat at time t for state i
                beta_hat[i, t] = sum_products / c[t]

        return beta_hat

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass


"""observations = [-0.2, 2.6, 1.3]

g1 = GaussD(means=[0], stdevs=[1])
g2 = GaussD(means=[3], stdevs=[2])

nStates = 2
nSamples = len(observations)
pX = np.zeros((nStates, nSamples))
scale_factors = np.zeros(nSamples)
for t in range(nSamples):
    for j, g in enumerate([g1, g2]):
        pX[j, t] = g.prob(observations[t])
    scale_factors[t] = pX[:, t].max()
    pX[:, t] /= scale_factors[t]
q = np.array([1, 0])
A = np.array([[0.9, 0.1, 0], [0, 0.9, 0.1]])
chain = MarkovChain(q, A)
alpha, c = chain.forward(pX)
print("Gaussian probabilities", pX)
print("final alpha ", alpha)
print("final c ", c)
beta = chain.backward(c, pX, True)
print("beta", beta)
h  = HMM( chain, [g1, g2])      
logprob = h.logprob(pX, scale_factors)
print("logprob", logprob)"""