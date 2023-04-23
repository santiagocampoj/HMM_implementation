import java.util.Arrays;

public class HiddenMarkovModel {

    private double[][] A;
    private double[][] B;
    private double[] pi;

    public HiddenMarkovModel(double[][] A, double[][] B, double[] pi) {
        this.A = A;
        this.B = B;
        this.pi = pi;
    }

    public double[][] forward(int[] O) {
        double[][] alpha = new double[O.length][A.length];
        for (int i = 0; i < A.length; i++) {
            alpha[0][i] = pi[i] * B[i][O[0]];
        }

        for (int t = 1; t < O.length; t++) {
            for (int j = 0; j < A.length; j++) {
                double sum = 0;
                for (int i = 0; i < A.length; i++) {
                    sum += alpha[t - 1][i] * A[i][j];
                }
                alpha[t][j] = sum * B[j][O[t]];
            }
        }

        return alpha;
    }

    public double[][] backward(int[] O) {
        double[][] beta = new double[O.length][A.length];
        Arrays.fill(beta[O.length - 1], 1);

        for (int t = O.length - 2; t >= 0; t--) {
            for (int i = 0; i < A.length; i++) {
                double sum = 0;
                for (int j = 0; j < A.length; j++) {
                    sum += beta[t + 1][j] * A[i][j] * B[j][O[t + 1]];
                }
                beta[t][i] = sum;
            }
        }

        return beta;
    }

    public int[] viterbi(int[] O) {
        double[][] delta = new double[O.length][A.length];
        int[][] psi = new int[O.length][A.length];

        for (int i = 0; i < A.length; i++) {
            delta[0][i] = pi[i] * B[i][O[0]];
        }

        for (int t = 1; t < O.length; t++) {
            for (int j = 0; j < A.length; j++) {
                double maxVal = Double.NEGATIVE_INFINITY;
                int maxIdx = 0;
                for (int i = 0; i < A.length; i++) {
                    double temp = delta[t - 1][i] * A[i][j];
                    if (temp > maxVal) {
                        maxVal = temp;
                        maxIdx = i;
                    }
                }
                delta[t][j] = maxVal * B[j][O[t]];
                psi[t][j] = maxIdx;
            }
        }

        int[] stateSeq = new int[O.length];
        double maxVal = Double.NEGATIVE_INFINITY;
        int maxIdx = 0;
        for (int i = 0; i < A.length; i++) {
            if (delta[O.length - 1][i] > maxVal) {
                maxVal = delta[O.length - 1][i];
                maxIdx = i;
            }
        }
        stateSeq[O.length - 1] = maxIdx;

        for (int t = O.length - 2; t >= 0; t--) {
            stateSeq[t] = psi[t +
