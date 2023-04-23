import numpy as np

class HiddenMarkovModel:
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def forward(self, O):
        # initialization
        alpha = np.zeros((len(O), len(self.A)))
        alpha[0, :] = self.pi * self.B[:, O[0]]

        # induction
        for t in range(1, len(O)):
            for j in range(len(self.A)):
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.A[:, j]) * self.B[j, O[t]]

        # termination
        P = np.sum(alpha[-1, :])

        return alpha, P

    def backward(self, O):
        # initialization
        beta = np.zeros((len(O), len(self.A)))
        beta[-1, :] = 1

        # induction
        for t in range(len(O) - 2, -1, -1):
            for i in range(len(self.A)):
                beta[t, i] = np.sum(beta[t + 1, :] * self.A[i, :] * self.B[:, O[t + 1]])

        # termination
        P = np.sum(beta[0, :] * self.pi * self.B[:, O[0]])

        return beta, P

    def viterbi(self, O):
        # initialization
        delta = np.zeros((len(O), len(self.A)))
        delta[0, :] = self.pi * self.B[:, O[0]]
        psi = np.zeros((len(O), len(self.A)), dtype=int)

        # induction
        for t in range(1, len(O)):
            for j in range(len(self.A)):
                delta[t, j] = np.max(delta[t - 1, :] * self.A[:, j]) * self.B[j, O[t]]
                psi[t, j] = np.argmax(delta[t - 1, :] * self.A[:, j])

        # termination
        state_seq = np.zeros(len(O), dtype=int)
        state_seq[-1] = np.argmax(delta[-1, :])

        for t in range(len(O) - 2, -1, -1):
            state_seq[t] = psi[t + 1, state_seq[t + 1]]
        
        return delta, state_seq



# state transition probability matrix (A)
A = np.array([
    [0.2, 0.3, 0.5], # Good mood
    [0.2, 0.2, 0.6], # Neutral mood
    [0.0, 0.2, 0.8]  # Bad mood
])

# observation probability matrix (B)
B = np.array([
    [0.8, 0.2], # Good mood - P(easy|good), P(difficult|good)
    [0.5, 0.5], # Neutral mood - P(easy|neutral), P(difficult|neutral)
    [0.1, 0.9]  # Bad mood - P(easy|bad), P(difficult|bad)
])

# initial state distribution (n)
pi = np.array([0.8, 0.2, 0.0]) # P(good), P(neutral), P(bad)

# HMM implementation
hmm = HiddenMarkovModel(A, B, pi)

# example observation sequence: ['easy', 'difficult', 'difficult', 'easy']
obs_seq = [0, 1, 1, 0]

# evaluation | Forward algorithm
alpha, likelihood = hmm.forward(obs_seq)
print(f"\nLikelihoof of the observation sequence: {likelihood}")

# decoding | Viterbi algorithm
delta, state_seq = hmm.viterbi(obs_seq)
print(f"\nMost likely sequence of hidden states (teacher's moods): {state_seq}")
