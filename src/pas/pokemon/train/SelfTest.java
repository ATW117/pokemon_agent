package src.pas.pokemon.train;


// SYSTEM IMPORTS
import java.io.File;
import java.util.List;
import java.util.Random;

import edu.bu.pas.pokemon.core.Battle;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.generators.BattleCreator;
import edu.bu.pas.pokemon.nn.Model;

// JAVA PROJECT IMPORTS
import src.pas.pokemon.agents.PolicyAgent;
import src.pas.pokemon.senses.CustomSensorArray;


/** Sanity checks for FastMLP <-> framework equivalence, gradients, and feature perspective. */
public class SelfTest
{
    public static void main(String[] args) throws Exception
    {
        java.io.PrintStream out = Evaluate.silenceStdout();
        Random rng = new Random(42);
        int n = CustomSensorArray.NUM_FEATURES;

        // 1) forward equivalence + save/load round trip through the framework format
        FastMLP fast = new FastMLP(n, PolicyAgent.HIDDEN1, PolicyAgent.HIDDEN2, rng);
        Model fw = new PolicyAgent().initModel();
        fast.toFramework(fw);
        double[][] X = new double[5][n];
        for (double[] row : X) for (int j = 0; j < n; j++) row[j] = rng.nextDouble() < 0.3 ? rng.nextGaussian() : 0.0;
        double[] a = fast.score(X), b = PolicyAgent.forwardRows(fw, X);
        double maxDiff = 0;
        for (int i = 0; i < a.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
        out.println("forward fast vs framework max|diff| = " + maxDiff);

        File tmp = File.createTempFile("selftest", ".model");
        fw.save(tmp.getPath());
        Model fw2 = new PolicyAgent().initModel();
        fw2.load(tmp.getPath());
        double[] c = PolicyAgent.forwardRows(fw2, X);
        maxDiff = 0;
        for (int i = 0; i < a.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - c[i]));
        out.println("save/load round trip max|diff| = " + maxDiff);
        tmp.delete();

        // 2) gradient check: one tiny SGD-like step should move the prediction toward the target
        FastMLP g = fast.copy();
        double[] y = new double[X.length];
        for (int i = 0; i < y.length; i++) y[i] = a[i] + 0.5;
        double before = g.trainStep(X, y, 0.0, 1e9, 0); // lr 0: just the loss
        for (int s = 0; s < 50; s++) g.trainStep(X, y, 1e-3, 1e9, 0);
        double after = g.trainStep(X, y, 0.0, 1e9, 0);
        out.println(String.format("loss before=%.5f after 50 Adam steps=%.5f (should drop)", before, after));

        // finite-difference check: analytic d(mean err^2 / 2)/dw vs numeric, on a few live weights
        FastMLP h = fast.copy();
        double[][] gW = new double[3][], gB = new double[3][];
        h.gradients(X, y, 1e9, gW, gB);
        double worst = 0;
        int[][] probes = new int[3][];
        for (int l = 0; l < 3; l++) {
            int arg = 0;
            for (int k = 1; k < gW[l].length; k++) if (Math.abs(gW[l][k]) > Math.abs(gW[l][arg])) arg = k;
            probes[l] = new int[] {l, arg};
        }
        for (int[] pr : probes) {
            int l = pr[0], idx = pr[1];
            double w0 = h.W[l][idx], e = 1e-6;
            h.W[l][idx] = w0 + e;
            double lp = h.gradients(X, y, 1e9, new double[3][], new double[3][]) / 2;
            h.W[l][idx] = w0 - e;
            double lm = h.gradients(X, y, 1e9, new double[3][], new double[3][]) / 2;
            h.W[l][idx] = w0;
            double num = (lp - lm) / (2 * e);
            double rel = Math.abs(num - gW[l][idx]) / Math.max(1e-8, Math.abs(num) + Math.abs(gW[l][idx]));
            worst = Math.max(worst, rel);
            out.println(String.format("  grad layer %d idx %d: analytic=%.6e numeric=%.6e", l, idx, gW[l][idx], num));
        }
        out.println("gradient check worst relative error = " + worst);

        // 3) perspective symmetry: team-0 view of a battle == team-1 view of the mirrored battle
        PolicyAgent p0 = new PolicyAgent();
        p0.initializeSenses(null);
        PolicyAgent p1 = new PolicyAgent();
        p1.initializeSenses(null);
        Battle battle = BattleCreator.makeRandomTeams(3, 3, 4, new Random(7), p0, p1);
        BattleView v = battle.getView();
        CustomSensorArray sens = new CustomSensorArray();
        List<MoveView> m0 = PolicyAgent.legalMovesFor(v, 0);
        List<MoveView> m1 = PolicyAgent.legalMovesFor(v, 1);
        double[] f0 = sens.encode(v, m0.get(0), 0);
        double[] f1 = sens.encode(v, m1.get(0), 1);
        out.println("team0 legal=" + m0.size() + " team1 legal=" + m1.size() + " features=" + f0.length
            + " (team0 and team1 encodings differ: " + !java.util.Arrays.equals(f0, f1) + ")");
        out.println("p0 team idx=" + p0.getMyTeamIdx() + " p1 team idx=" + p1.getMyTeamIdx());

        // switch actions to different slots must encode differently
        MoveView sw1 = null, sw2 = null;
        for (MoveView mv : m0) {
            if (mv instanceof edu.bu.pas.pokemon.core.SwitchMove.SwitchMoveView) {
                if (sw1 == null) sw1 = mv; else if (sw2 == null) sw2 = mv;
            }
        }
        if (sw1 != null && sw2 != null) {
            out.println("two switch targets encode differently: "
                + !java.util.Arrays.equals(sens.encode(v, sw1, 0), sens.encode(v, sw2, 0)));
        }
    }
}
