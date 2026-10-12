package src.pas.pokemon.agents;


// SYSTEM IMPORTS
import net.sourceforge.argparse4j.inf.Namespace;

import edu.bu.pas.pokemon.core.Battle;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.Pokemon;
import edu.bu.pas.pokemon.core.StruggleMove;
import edu.bu.pas.pokemon.core.SwitchMove;
import edu.bu.pas.pokemon.core.Team;
import edu.bu.pas.pokemon.linalg.Matrix;
import edu.bu.pas.pokemon.nn.Model;
import edu.bu.pas.pokemon.nn.Parameter;
import edu.bu.pas.pokemon.utils.Pair;

import java.util.List;

// JAVA PROJECT IMPORTS
import src.pas.pokemon.rewards.CustomRewardFunction;


/**
 * PolicyAgent plus a one-turn lookahead at decision time (no extra training needed).
 *
 * For each of our candidate moves a and each opponent reply b, the turn is simulated `samples` times on
 * a copy of the battle (the engine rolls accuracy, damage, crits, status, ...), faint replacements are
 * filled in with the replacement heuristic, and the result is scored as
 *     r(s, a, s') + gamma * max_a' Q(s', a')
 * The opponent is modelled as a mixture: with weight `oppGreedy` it plays the move our own network would
 * pick in its place, otherwise it plays uniformly at random. We play the move with the best expected score.
 *
 * Tunables (system properties): pokemon.lookahead.samples (4), pokemon.lookahead.oppGreedy (0.5),
 * pokemon.lookahead.gamma (0.97).
 */
public class LookaheadAgent
    extends PolicyAgent
{
    private final int samples = Integer.getInteger("pokemon.lookahead.samples", 4);
    private final double oppGreedy = Double.parseDouble(System.getProperty("pokemon.lookahead.oppGreedy", "0.5"));
    private final double gamma = Double.parseDouble(System.getProperty("pokemon.lookahead.gamma", "0.97"));

    @Override
    public void initialize(Namespace args)
    {
        super.initialize(args);
        // the framework's Matrix matmul is far too slow for thousands of evaluations per turn
        this.setScorer(new ArrayScorer(this.getModel()));
    }

    @Override
    public MoveView getMove(BattleView view)
    {
        List<MoveView> mine = this.candidateMoves(view);
        if (mine.isEmpty()) {
            return null;
        }
        if (mine.size() == 1) {
            this.recordChoice(mine.get(0));
            return mine.get(0);
        }
        if (this.getScorer() == null) {
            this.setScorer(new ArrayScorer(this.getModel()));
        }
        int me = this.getMyTeamIdx() == 1 ? 1 : 0;
        int opp = 1 - me;

        // opponent model: greedy-by-our-network mixed with uniform
        List<MoveView> theirs = legalMovesFor(view, opp);
        double[] pOpp = new double[Math.max(1, theirs.size())];
        if (theirs.isEmpty()) {
            pOpp[0] = 1.0;
        } else {
            double[] qOpp = this.getScorer().score(this.encodeAllFor(view, theirs, opp));
            int greedy = argmax(qOpp);
            for (int j = 0; j < theirs.size(); j++) {
                pOpp[j] = (1.0 - this.oppGreedy) / theirs.size() + (j == greedy ? this.oppGreedy : 0.0);
            }
        }

        double[] score = new double[mine.size()];
        for (int i = 0; i < mine.size(); i++) {
            for (int j = 0; j < pOpp.length; j++) {
                MoveView b = theirs.isEmpty() ? null : theirs.get(j);
                double total = 0.0;
                for (int k = 0; k < this.samples; k++) {
                    total += this.simulate(view, me, mine.get(i), b);
                }
                score[i] += pOpp[j] * total / this.samples;
            }
        }
        MoveView chosen = mine.get(argmax(score));
        this.recordChoice(chosen);
        return chosen;
    }

    /** One sampled turn: r(s, a, s') + gamma * max_a' Q(s', a'). */
    private double simulate(BattleView view, int me, MoveView a, MoveView b)
    {
        Battle sim = new Battle(view);
        // Team(TeamView) adds its pokemon after the loop that links them to the team, leaving
        // pokemon.getTeam() null (the engine NPEs in canApplyMove and ignores screens in damage)
        for (int t = 0; t < 2; t++) {
            for (Pokemon p : sim.getTeam(t).getPokemon()) {
                p.setTeam(sim.getTeam(t));
            }
        }
        Move ma = toMove(sim, me, a);
        Move mb = b == null ? null : toMove(sim, 1 - me, b);
        Pair<Move, Move> pair = me == 0 ? new Pair<Move, Move>(ma, mb) : new Pair<Move, Move>(mb, ma);
        sim.applyMoves(pair);
        sim.applyPostTurnConditions();
        // the copied teams have no agents, so the engine skips faint replacements; fill them in ourselves
        for (int t = 0; t < 2; t++) {
            Team team = sim.getTeam(t);
            if (!sim.isOver() && team.getActivePokemon().hasFainted()) {
                Integer idx = chooseReplacement(sim.getView(), t);
                if (idx != null) {
                    team.switchActivePokemonTo(idx);
                }
            }
        }
        BattleView s2 = sim.getView();
        double value = CustomRewardFunction.reward(view, a, s2, me);
        if (!s2.isOver()) {
            List<MoveView> next = legalMovesFor(s2, me);
            if (!next.isEmpty()) {
                double[] q = this.getScorer().score(this.encodeAllFor(s2, next, me));
                value += this.gamma * q[argmax(q)];
            }
        }
        return value;
    }

    /** Mirrors Battle.getMoves: the active pokemon's equipped move by name, its locked-in move, Struggle, or a switch. */
    private static Move toMove(Battle sim, int team, MoveView mv)
    {
        Pokemon active = sim.getTeam(team).getActivePokemon();
        if (active.getActiveMove() != null) {
            return active.getActiveMove();
        }
        if (mv instanceof SwitchMove.SwitchMoveView) {
            return new SwitchMove((SwitchMove.SwitchMoveView) mv);
        }
        for (Move m : active.getMoves()) {
            if (m != null && m.getName().equals(mv.getName())) {
                return m;
            }
        }
        return new StruggleMove();
    }

    private static int argmax(double[] v)
    {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) best = i;
        }
        return best;
    }

    /** Forward-only copy of the framework Sequential (Dense/ReLU/.../Dense) in plain arrays. */
    static final class ArrayScorer
        implements QScorer
    {
        private final double[][] W;
        private final double[][] b;
        private final int[] in;
        private final int[] out;

        ArrayScorer(Model model)
        {
            List<Parameter> params = model.getParameters();
            int layers = params.size() / 2;
            W = new double[layers][];
            b = new double[layers][];
            in = new int[layers];
            out = new int[layers];
            for (int l = 0; l < layers; l++) {
                Matrix w = params.get(2 * l).getValue();
                Matrix bb = params.get(2 * l + 1).getValue();
                in[l] = w.getShape().getNumRows();
                out[l] = w.getShape().getNumCols();
                W[l] = new double[in[l] * out[l]];
                b[l] = new double[out[l]];
                for (int i = 0; i < in[l]; i++) {
                    for (int j = 0; j < out[l]; j++) {
                        W[l][i * out[l] + j] = w.get(i, j);
                    }
                }
                for (int j = 0; j < out[l]; j++) {
                    b[l][j] = bb.get(0, j);
                }
            }
        }

        @Override
        public double[] score(double[][] feats)
        {
            double[] q = new double[feats.length];
            for (int r = 0; r < feats.length; r++) {
                double[] x = feats[r];
                for (int l = 0; l < W.length; l++) {
                    double[] y = b[l].clone();
                    for (int i = 0; i < in[l]; i++) {
                        double xi = x[i];
                        if (xi == 0.0) continue;
                        int base = i * out[l];
                        for (int j = 0; j < out[l]; j++) {
                            y[j] += xi * W[l][base + j];
                        }
                    }
                    if (l < W.length - 1) {
                        for (int j = 0; j < y.length; j++) {
                            if (y[j] < 0) y[j] = 0;
                        }
                    }
                    x = y;
                }
                q[r] = x[0];
            }
            return q;
        }
    }
}
