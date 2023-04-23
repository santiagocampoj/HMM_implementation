# HMM implementation

"""HMM implementation
Problem definition:

– A teacher assigns the students two different kinds of homework:
• Easy exercises that can be solved in 5 minutes
• Difficult exercises that are solved in 1 hour

– The assignation of homework depends on the mood of the teacher:
• Good mood: 80% of times an easy work is assigned and 20% a difficult one
• Neutral mood: 50% of times the assigned work is easy and 50% of times difficult
• Bad mood: 10% of times the work assigned is easy and 90% of times difficult

– The teacher hides his/her mood, but we know that:
• If one day he/she is in a good mood, the following day there is a 20% chance the mood does
not change, a 30% chance that the mood is neutral and 50% chance that it is bad
• If one day he/she is in a bad mood, the following day there is a 80% chance the mood does
not change and 20% chance that the mood is neutral
• Finally, if the mood is neutral there is a 20% chance of continuing with the same mood the
following day, 20% chance of being in a good mood and 60% chance of being in a bad mood

– The students only know the kind of homework assigned
– At the beginning of the course the teacher returns from holidays, 80% of
times in a good mood and 20% of times in neutral mood
"""

import numpy as np

class HiddenMarkovModel:
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def forward(self, O):
        # Initialization
        alpha = np.zeros((len(O), len(self.A)))
        alpha[0, :] = self.pi * self.B[:, O[0]]

        # Induction
        for t in range(1, len(O)):
            for j in range(len(self.A)):
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.A[:, j]) * self.B[j, O[t]]

        # Termination
        P = np.sum(alpha[-1, :])

        return alpha, P

    def backward(self, O):
        # Initialization
        beta = np.zeros((len(O), len(self.A)))
        beta[-1, :] = 1

        # Induction
        for t in range(len(O) - 2, -1, -1):
            for i in range(len(self.A)):
                beta[t, i] = np.sum(beta[t + 1, :] * self.A[i, :] * self.B[:, O[t + 1]])

        # Termination
        P = np.sum(beta[0, :] * self.pi * self.B[:, O[0]])

        return beta, P

    def viterbi(self, O):
        # Initialization
        delta = np.zeros((len(O), len(self.A)))
        delta[0, :] = self.pi * self.B[:, O[0]]
        psi = np.zeros((len(O), len(self.A)), dtype=int)

        # Induction
        for t in range(1, len(O)):
            for j in range(len(self.A)):
                delta[t, j] = np.max(delta[t - 1, :] * self.A[:, j]) * self.B[j, O[t]]
                psi[t, j] = np.argmax(delta[t - 1, :] * self.A[:, j])

        # Termination
        state_seq = np.zeros(len(O), dtype=int)
        state_seq[-1] = np.argmax(delta[-1, :])

        for t in range(len(O) - 2, -1, -1):
            state_seq[t] = psi[t + 1, state_seq[t + 1]]
        
        return delta, state_seq



# State transition probability matrix (A)
A = np.array([
    [0.2, 0.3, 0.5], # Good mood
    [0.2, 0.2, 0.6], # Neutral mood
    [0.0, 0.2, 0.8]  # Bad mood
])

# Observation probability matrix (B)
B = np.array([
    [0.8, 0.2], # Good mood - P(easy|good), P(difficult|good)
    [0.5, 0.5], # Neutral mood - P(easy|neutral), P(difficult|neutral)
    [0.1, 0.9]  # Bad mood - P(easy|bad), P(difficult|bad)
])

# Initial state distribution (n)
pi = np.array([0.8, 0.2, 0.0]) # P(good), P(neutral), P(bad)

# HMM implementation
hmm = HiddenMarkovModel(A, B, pi)

# Example observation sequence: ['easy', 'difficult', 'difficult', 'easy']
obs_seq = [0, 1, 1, 0]

# Evaluation | Forward algorithm
alpha, likelihood = hmm.forward(obs_seq)
print(f"\nLikelihoof of the observation sequence: {likelihood}")

# Decoding | Viterbi algorithm
delta, state_seq = hmm.viterbi(obs_seq)
print(f"\nMost likely sequence of hidden states (teacher's moods): {state_seq}")