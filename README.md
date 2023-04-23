# HMM_implementation
This code is an implementation of a Hidden Markov Model (HMM) to better understand an exercise scenario. The problem statement is as follows:

- There is a teacher who assigns students two different kinds of homework:
  1. Easy exercises that can be solved in 5 minutes
  2. Difficult exercises that are solved in 1 hour

- The assignment of homework depends on the mood of the teacher:
  * Good mood: 80% of times an easy work is assigned and 20% a difficult one
  * Neutral mood: 50% of times the assigned work is easy and 50% of times difficult
  * Bad mood: 10% of times the work assigned is easy and 90% of times difficult

- The teacher's mood is hidden, but we know the following:
  * If one day the teacher is in a good mood, the following day there is a 20% chance the mood does not change, a 30% chance that the mood is neutral, and 50% chance that it is bad.
  * If one day the teacher is in a bad mood, the following day there is an 80% chance the mood does not change and 20% chance that the mood is neutral.
  * Finally, if the mood is neutral, there is a 20% chance of continuing with the same mood the following day, 20% chance of being in a good mood, and 60% chance of being in a bad mood.

- The students only know the kind of homework assigned and not the teacher's mood.
- At the beginning of the course, the teacher returns from holidays, 80% of times in a good mood, and 20% of times in a neutral mood.

=========================================
=========================================
=========================================

The code implements a Hidden Markov Model (HMM) to solve this problem. It defines a class, `HiddenMarkovModel`, which takes the state transition probability matrix (A), the observation probability matrix (B), and the initial state distribution (pi) as inputs. The class has three methods:

1. `forward`: Computes the forward algorithm for a given observation sequence.
2. `backward`: Computes the backward algorithm for a given observation sequence.
3. `viterbi`: Computes the Viterbi algorithm for a given observation sequence, returning the most likely sequence of hidden states (teacher's moods).

The problem is solved using the implemented HMM by providing an example observation sequence (e.g., ['easy', 'difficult', 'difficult', 'easy']). The HMM calculates the likelihood of the observation sequence and the most likely sequence of hidden states (teacher's moods) using the forward and Viterbi algorithms, respectively.

By understanding and implementing the HMM for this problem, we konw how the teacher's moods affect the assignment of homework and how students might predict their teacher's mood based on the assigned homework.

