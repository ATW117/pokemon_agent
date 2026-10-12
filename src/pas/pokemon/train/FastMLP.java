package src.pas.pokemon.train;


// SYSTEM IMPORTS
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import edu.bu.pas.pokemon.linalg.Matrix;
import edu.bu.pas.pokemon.nn.Model;
import edu.bu.pas.pokemon.nn.Parameter;

// JAVA PROJECT IMPORTS
import src.pas.pokemon.agents.PolicyAgent;


/**
 * Plain-array ReLU MLP (in -> h1 -> h2 -> 1) with Adam, used only for training speed.
 * Weights are laid out exactly like the framework's Dense layers (W is in x out, row-major),
 * so toFramework/fromFramework are straight copies and the saved params.model is unchanged in format.
 */
public class FastMLP
    implements PolicyAgent.QScorer
{
    final int[] sizes;      // {in, h1, h2, 1}
    final double[][] W;     // W[l][i * out + j]
    final double[][] b;

    // Adam state
    private double[][] mW, vW, mB, vB;
    private long t = 0;

    public FastMLP(int in, int h1, int h2, Random rng)
    {
        this.sizes = new int[] {in, h1, h2, 1};
        this.W = new double[3][];
        this.b = new double[3][];
        for (int l = 0; l < 3; l++) {
            int nIn = sizes[l], nOut = sizes[l + 1];
            W[l] = new double[nIn * nOut];
            b[l] = new double[nOut];
            double std = Math.sqrt(2.0 / nIn);
            for (int k = 0; k < W[l].length; k++) {
                W[l][k] = rng.nextGaussian() * std;
            }
        }
    }

    public FastMLP copy()
    {
        FastMLP c = new FastMLP(sizes[0], sizes[1], sizes[2], new Random(0));
        c.copyFrom(this);
        return c;
    }

    public void copyFrom(FastMLP o)
    {
        for (int l = 0; l < 3; l++) {
            System.arraycopy(o.W[l], 0, W[l], 0, W[l].length);
            System.arraycopy(o.b[l], 0, b[l], 0, b[l].length);
        }
    }

    // ------------------------------------------------------------------ inference

    public double forwardOne(double[] x)
    {
        double[] h1 = layer(0, x, true);
        double[] h2 = layer(1, h1, true);
        return layer(2, h2, false)[0];
    }

    private double[] layer(int l, double[] x, boolean relu)
    {
        int nIn = sizes[l], nOut = sizes[l + 1];
        double[] w = W[l];
        double[] out = b[l].clone();
        for (int i = 0; i < nIn; i++) {
            double xi = x[i];
            if (xi == 0.0) continue; // inputs are mostly one-hot zeros
            int base = i * nOut;
            for (int j = 0; j < nOut; j++) {
                out[j] += xi * w[base + j];
            }
        }
        if (relu) {
            for (int j = 0; j < nOut; j++) {
                if (out[j] < 0) out[j] = 0;
            }
        }
        return out;
    }

    @Override
    public double[] score(double[][] feats)
    {
        double[] out = new double[feats.length];
        for (int r = 0; r < feats.length; r++) {
            out[r] = forwardOne(feats[r]);
        }
        return out;
    }

    /** Batch scoring spread over the common fork-join pool (used for target computation). */
    public double[] scoreParallel(double[][] feats)
    {
        double[] out = new double[feats.length];
        IntStream.range(0, feats.length).parallel().forEach(r -> out[r] = forwardOne(feats[r]));
        return out;
    }

    // ------------------------------------------------------------------ training

    /**
     * One Adam step on mean squared error against targets. The per-sample error is clipped to
     * [-huber, huber], which gives the Huber-loss gradient.
     * @return mean squared (unclipped) TD error
     */
    public double trainStep(double[][] X, double[] y, double lr, double huber, double gradClipNorm)
    {
        double[][] gW = new double[3][];
        double[][] gB = new double[3][];
        double loss = gradients(X, y, huber, gW, gB);

        // global-norm gradient clipping
        double norm = 0.0;
        for (int l = 0; l < 3; l++) {
            for (double g : gW[l]) norm += g * g;
            for (double g : gB[l]) norm += g * g;
        }
        norm = Math.sqrt(norm);
        double scale = (gradClipNorm > 0 && norm > gradClipNorm) ? gradClipNorm / norm : 1.0;

        if (lr > 0) {
            adam(gW, gB, scale, lr);
        }
        return loss;
    }

    /** Fills gW/gB with d(loss)/d(param), where loss = mean(err^2)/2 (Huber-clipped). Returns mean err^2. */
    double gradients(double[][] X, double[] y, double huber, double[][] gW, double[][] gB)
    {
        final int B = X.length;
        for (int l = 0; l < 3; l++) {
            gW[l] = new double[W[l].length];
            gB[l] = new double[b[l].length];
        }
        double sqErr = 0.0;
        int n0 = sizes[0], n1 = sizes[1], n2 = sizes[2];
        double[] d2 = new double[n2];
        double[] d1 = new double[n1];
        for (int r = 0; r < B; r++) {
            double[] x = X[r];
            double[] a1 = layer(0, x, true);
            double[] a2 = layer(1, a1, true);
            double q = layer(2, a2, false)[0];
            double err = q - y[r];
            sqErr += err * err;
            double d3 = Math.max(-huber, Math.min(huber, err)) / B;

            // output layer
            for (int j = 0; j < n2; j++) {
                gW[2][j] += a2[j] * d3;
                d2[j] = a2[j] > 0 ? W[2][j] * d3 : 0.0;
            }
            gB[2][0] += d3;

            // hidden 2
            for (int i = 0; i < n1; i++) {
                double ai = a1[i];
                int base = i * n2;
                double acc = 0.0;
                for (int j = 0; j < n2; j++) {
                    if (ai != 0.0) gW[1][base + j] += ai * d2[j];
                    acc += W[1][base + j] * d2[j];
                }
                d1[i] = ai > 0 ? acc : 0.0;
            }
            for (int j = 0; j < n2; j++) gB[1][j] += d2[j];

            // hidden 1
            for (int i = 0; i < n0; i++) {
                double xi = x[i];
                if (xi == 0.0) continue;
                int base = i * n1;
                for (int j = 0; j < n1; j++) {
                    gW[0][base + j] += xi * d1[j];
                }
            }
            for (int j = 0; j < n1; j++) gB[0][j] += d1[j];
        }
        return sqErr / B;
    }

    private void adam(double[][] gW, double[][] gB, double scale, double lr)
    {
        final double b1 = 0.9, b2 = 0.999, eps = 1e-8;
        if (mW == null) {
            mW = new double[3][]; vW = new double[3][]; mB = new double[3][]; vB = new double[3][];
            for (int l = 0; l < 3; l++) {
                mW[l] = new double[W[l].length]; vW[l] = new double[W[l].length];
                mB[l] = new double[b[l].length]; vB[l] = new double[b[l].length];
            }
        }
        t++;
        double c1 = 1.0 - Math.pow(b1, t), c2 = 1.0 - Math.pow(b2, t);
        for (int l = 0; l < 3; l++) {
            adamUpdate(W[l], gW[l], mW[l], vW[l], scale, lr, b1, b2, eps, c1, c2);
            adamUpdate(b[l], gB[l], mB[l], vB[l], scale, lr, b1, b2, eps, c1, c2);
        }
    }

    private static void adamUpdate(double[] p, double[] g, double[] m, double[] v, double scale, double lr,
                                   double b1, double b2, double eps, double c1, double c2)
    {
        for (int k = 0; k < p.length; k++) {
            double gk = g[k] * scale;
            m[k] = b1 * m[k] + (1 - b1) * gk;
            v[k] = b2 * v[k] + (1 - b2) * gk * gk;
            p[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + eps);
        }
    }

    // ------------------------------------------------------------------ framework interop

    /** Copies weights into a framework Sequential built by PolicyAgent.initModel(). */
    public void toFramework(Model model)
    {
        List<Parameter> params = model.getParameters();
        for (int l = 0; l < 3; l++) {
            Matrix w = params.get(2 * l).getValue();
            Matrix bb = params.get(2 * l + 1).getValue();
            int nIn = sizes[l], nOut = sizes[l + 1];
            checkShape(w, nIn, nOut);
            for (int i = 0; i < nIn; i++) {
                for (int j = 0; j < nOut; j++) {
                    w.set(i, j, W[l][i * nOut + j]);
                }
            }
            for (int j = 0; j < nOut; j++) {
                bb.set(0, j, b[l][j]);
            }
        }
    }

    public void fromFramework(Model model)
    {
        List<Parameter> params = model.getParameters();
        for (int l = 0; l < 3; l++) {
            Matrix w = params.get(2 * l).getValue();
            Matrix bb = params.get(2 * l + 1).getValue();
            int nIn = sizes[l], nOut = sizes[l + 1];
            checkShape(w, nIn, nOut);
            for (int i = 0; i < nIn; i++) {
                for (int j = 0; j < nOut; j++) {
                    W[l][i * nOut + j] = w.get(i, j);
                }
            }
            for (int j = 0; j < nOut; j++) {
                b[l][j] = bb.get(0, j);
            }
        }
    }

    private static void checkShape(Matrix w, int nIn, int nOut)
    {
        if (w.getShape().getNumRows() != nIn || w.getShape().getNumCols() != nOut) {
            throw new IllegalStateException("FastMLP: framework layer shape " + w.getShape() + " != (" + nIn + "," + nOut + ")");
        }
    }
}
