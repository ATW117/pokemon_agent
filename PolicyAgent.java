package src.pas.pokemon.agents;


// SYSTEM IMPORTS
import net.sourceforge.argparse4j.inf.Namespace;

import edu.bu.pas.pokemon.agents.NeuralQAgent;
import edu.bu.pas.pokemon.agents.rewards.RewardFunction;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.SwitchMove;
import edu.bu.pas.pokemon.core.Team.TeamView;
import edu.bu.pas.pokemon.core.Pokemon.PokemonView;
import edu.bu.pas.pokemon.core.enums.Flag;
import edu.bu.pas.pokemon.core.enums.NonVolatileStatus;
import edu.bu.pas.pokemon.core.enums.Stat;
import edu.bu.pas.pokemon.linalg.Matrix;
import edu.bu.pas.pokemon.nn.Model;
import edu.bu.pas.pokemon.nn.Parameter;
import edu.bu.pas.pokemon.nn.layers.Dense;
import edu.bu.pas.pokemon.nn.layers.ReLU;
import edu.bu.pas.pokemon.nn.models.Sequential;
import edu.bu.pas.pokemon.training.data.ReplayBuffer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

// JAVA PROJECT IMPORTS
import src.pas.pokemon.rewards.CustomRewardFunction;
import src.pas.pokemon.senses.CustomSensorArray;


/**
 * Q(s, a) agent. Training and evaluation choose from the same legal action set (attacks + switches);
 * the only difference is epsilon-greedy exploration during training.
 *
 * Two training paths are supported:
 *   - the course's ParallelTrain/Train (uses makeGroundTruth below, Double DQN with a target network)
 *   - src.pas.pokemon.train.DQNTrainer (own loop, injects a fast scorer and sets epsilon directly)
 */
public class PolicyAgent
    extends NeuralQAgent
{
    public static final int HIDDEN1 = 512;
    public static final int HIDDEN2 = 256;

    /** Lets the trainer plug in a faster network implementation; rows of feats are (state, action) encodings. */
    public interface QScorer
    {
        double[] score(double[][] feats);
    }

    // epsilon schedule for the framework training path (shared by all instances in the JVM)
    private static final AtomicLong GLOBAL_STEPS = new AtomicLong();
    private static final double EPS_START = Double.parseDouble(System.getProperty("pokemon.eps.start", "1.0"));
    private static final double EPS_END = Double.parseDouble(System.getProperty("pokemon.eps.end", "0.05"));
    private static final long EPS_DECAY = Long.parseLong(System.getProperty("pokemon.eps.decay", "2000000"));

    private final Random random = new Random();
    private boolean isTraining = false;
    private Double epsilonOverride = null;
    private QScorer scorer = null;
    private Model targetNetwork = null;
    private int consecutiveSwitches = 0;

    public PolicyAgent()
    {
        super();
    }

    // ------------------------------------------------------------------ setup

    public void initializeSenses(Namespace args)
    {
        this.setSensorArray(new CustomSensorArray(() -> this.getMyTeamIdx() == 1 ? 1 : 0));
    }

    @Override
    public void initialize(Namespace args)
    {
        // builds the model via initModel() and loads params when --inFile is given
        super.initialize(args);
        this.initializeSenses(args);
    }

    @Override
    public Model initModel()
    {
        Sequential q = new Sequential();
        q.add(new Dense(CustomSensorArray.NUM_FEATURES, HIDDEN1));
        q.add(new ReLU());
        q.add(new Dense(HIDDEN1, HIDDEN2));
        q.add(new ReLU());
        q.add(new Dense(HIDDEN2, 1));
        heInit(q, new Random());
        return q;
    }

    /** Matrix.randn gives std=1 weights, which saturates a 300-input network; rescale to He init with zero bias. */
    public static void heInit(Model model, Random rng)
    {
        List<Parameter> params = model.getParameters();
        for (int p = 0; p + 1 < params.size(); p += 2) {
            Matrix W = params.get(p).getValue();
            Matrix b = params.get(p + 1).getValue();
            int fanIn = W.getShape().getNumRows();
            double std = Math.sqrt(2.0 / fanIn);
            for (int i = 0; i < fanIn; i++) {
                for (int j = 0; j < W.getShape().getNumCols(); j++) {
                    W.set(i, j, rng.nextGaussian() * std);
                }
            }
            for (int j = 0; j < b.getShape().getNumCols(); j++) {
                b.set(0, j, 0.0);
            }
        }
    }

    /** Share another agent's network (trainer gives each worker its own agent but one set of weights). */
    public void useModel(Model model, QScorer scorer)
    {
        this.setModel(model);
        this.scorer = scorer;
        if (this.getSensorArray() == null) {
            this.initializeSenses(null);
        }
    }

    public void setExplorationEpsilon(Double eps)
    {
        this.epsilonOverride = eps;
    }

    public void setTrainingMode(boolean training)
    {
        this.isTraining = training;
    }

    @Override
    public void train()
    {
        this.setTrainingMode(true);
    }

    @Override
    public void eval()
    {
        this.setTrainingMode(false);
    }

    private double currentEpsilon()
    {
        if (this.epsilonOverride != null) {
            return this.epsilonOverride;
        }
        if (!this.isTraining) {
            return 0.0;
        }
        double progress = Math.min(1.0, GLOBAL_STEPS.get() / (double) EPS_DECAY);
        return EPS_END + (EPS_START - EPS_END) * (1.0 - progress);
    }

    // ------------------------------------------------------------------ acting

    /** Legal actions for whoever this agent is in the given state. */
    public List<MoveView> legalMoves(BattleView view)
    {
        return legalMovesFor(view, this.getMyTeamIdx() == 1 ? 1 : 0);
    }

    public static List<MoveView> legalMovesFor(BattleView view, int teamIdx)
    {
        List<MoveView> moves = new ArrayList<>();
        if (view == null || view.isOver()) {
            return moves;
        }
        TeamView team = view.getTeamView(teamIdx);
        PokemonView active = team.getActivePokemonView();
        if (active == null) {
            return moves;
        }
        if (!active.hasFainted()) {
            moves.addAll(active.getAvailableMoves());
            // locked into a multi-turn move (the engine ignores whatever we submit) or trapped: no switching
            if (active.getActiveMoveView() != null || active.getFlag(Flag.TRAPPED)) {
                return moves;
            }
        }
        for (int i = 0; i < team.size(); i++) {
            PokemonView p = team.getPokemonView(i);
            if (i != team.getActivePokemonIdx() && p != null && !p.hasFainted()) {
                moves.add(new SwitchMove(i).getView());
            }
        }
        return moves;
    }

    public static final int MAX_CONSECUTIVE_SWITCHES = 2;

    @Override
    public MoveView getMove(BattleView view)
    {
        List<MoveView> moves = this.legalMoves(view);
        if (moves.isEmpty()) {
            return null;
        }
        // a greedy policy can ping-pong between two switches forever; after a few in a row, force an attack
        if (this.consecutiveSwitches >= MAX_CONSECUTIVE_SWITCHES) {
            List<MoveView> attacks = new ArrayList<>();
            for (MoveView m : moves) {
                if (!(m instanceof SwitchMove.SwitchMoveView)) attacks.add(m);
            }
            if (!attacks.isEmpty()) {
                moves = attacks;
            }
        }
        if (this.isTraining && this.epsilonOverride == null) {
            GLOBAL_STEPS.incrementAndGet();
        }
        MoveView chosen;
        double eps = this.currentEpsilon();
        if (eps > 0.0 && this.random.nextDouble() < eps) {
            chosen = moves.get(this.random.nextInt(moves.size()));
        } else {
            chosen = this.argmaxOver(view, moves);
        }
        this.consecutiveSwitches = (chosen instanceof SwitchMove.SwitchMoveView) ? this.consecutiveSwitches + 1 : 0;
        return chosen;
    }

    @Override
    public MoveView argmax(BattleView view)
    {
        return this.argmaxOver(view, this.legalMoves(view));
    }

    /** Some framework entry points (e.g. RandomBattle) never call initialize(); build what we need lazily. */
    private void ensureReady()
    {
        if (this.getSensorArray() == null) {
            this.initializeSenses(null);
        }
        if (this.getModel() == null && this.scorer == null) {
            this.setModel(this.initModel());
        }
    }

    private MoveView argmaxOver(BattleView view, List<MoveView> moves)
    {
        if (moves.isEmpty()) {
            return null;
        }
        this.ensureReady();
        double[] q = this.qValues(this.getModel(), this.scorer, view, moves);
        int best = 0;
        for (int i = 1; i < q.length; i++) {
            if (q[i] > q[best]) best = i;
        }
        return moves.get(best);
    }

    public double[][] encodeAll(BattleView view, List<MoveView> moves)
    {
        this.ensureReady();
        CustomSensorArray sensors = (CustomSensorArray) this.getSensorArray();
        int me = this.getMyTeamIdx() == 1 ? 1 : 0;
        double[][] feats = new double[moves.size()][];
        for (int i = 0; i < moves.size(); i++) {
            feats[i] = sensors.encode(view, moves.get(i), me);
        }
        return feats;
    }

    /** Q-values for each move, batched into one forward pass. */
    public double[] qValues(Model model, QScorer fast, BattleView view, List<MoveView> moves)
    {
        double[][] feats = this.encodeAll(view, moves);
        if (fast != null) {
            return fast.score(feats);
        }
        return forwardRows(model, feats);
    }

    public static double[] forwardRows(Model model, double[][] feats)
    {
        double[] out = new double[feats.length];
        if (feats.length == 0) {
            return out;
        }
        try {
            Matrix X = Matrix.zeros(feats.length, feats[0].length);
            for (int i = 0; i < feats.length; i++) {
                for (int j = 0; j < feats[i].length; j++) {
                    X.set(i, j, feats[i][j]);
                }
            }
            Matrix Y = model.forward(X);
            for (int i = 0; i < feats.length; i++) {
                out[i] = Y.get(i, 0);
            }
        } catch (Exception e) {
            throw new RuntimeException("PolicyAgent: forward pass failed", e);
        }
        return out;
    }

    // ------------------------------------------------------------------ replacement after a faint

    @Override
    public Integer chooseNextPokemon(BattleView view)
    {
        int me = this.getMyTeamIdx() == 1 ? 1 : 0;
        TeamView myTeam = view.getTeamView(me);
        TeamView oppTeam = view.getTeamView(1 - me);
        PokemonView opp = oppTeam.getActivePokemonView();

        Integer bestIdx = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < myTeam.size(); i++) {
            PokemonView p = myTeam.getPokemonView(i);
            if (p == null || p.hasFainted()) {
                continue;
            }
            double hp = p.getCurrentStat(Stat.HP) / (double) Math.max(1, p.getInitialStat(Stat.HP));
            double score = 0.8 * hp;
            if (opp != null && !opp.hasFainted()) {
                double dealt = Math.min(1.5, CustomSensorArray.bestDamageFrac(p, opp, oppTeam));
                double taken = Math.min(1.5, CustomSensorArray.bestDamageFrac(opp, p, myTeam));
                boolean faster = CustomSensorArray.effectiveSpeed(p) > CustomSensorArray.effectiveSpeed(opp);
                score += dealt - taken;
                // outspeeding and KOing first is decisive
                if (faster && dealt >= 1.0) score += 0.75;
                if (!faster && taken >= 1.0) score -= 0.75;
            }
            score -= statusPenalty(p.getNonVolatileStatus());
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private static double statusPenalty(NonVolatileStatus status)
    {
        if (status == null) return 0.0;
        switch (status) {
            case SLEEP:
            case FREEZE:
                return 0.5;
            case PARALYSIS:
                return 0.3;
            case TOXIC:
                return 0.25;
            case BURN:
            case POISON:
                return 0.15;
            default:
                return 0.0;
        }
    }

    // ------------------------------------------------------------------ framework training path

    /**
     * Double DQN ground truth for the course's Train/ParallelTrain:
     *   y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
     * Rewards are computed from this agent's perspective. Note that the framework's TrainingBattleDriver
     * pairs each action with the following turn's transition; prefer DQNTrainer, which records (s, a, s')
     * correctly.
     */
    @Override
    public Matrix makeGroundTruth(ReplayBuffer buffer, double discountFactor, RewardFunction rewardFunction)
    {
        if (this.targetNetwork == null) {
            this.targetNetwork = this.initModel();
        }
        // the framework calls this once per cycle, so the target is the network as of the start of the cycle
        copyParams(this.getModel(), this.targetNetwork);
        int me = this.getMyTeamIdx() == 1 ? 1 : 0;
        Matrix y = Matrix.zeros(buffer.size(), 1);
        for (int i = 0; i < buffer.size(); i++) {
            BattleView s = buffer.getState(i);
            MoveView a = buffer.getAction(i);
            BattleView s2 = buffer.getNextState(i);
            double target = CustomRewardFunction.reward(s, a, s2, me);
            List<MoveView> next = this.legalMoves(s2);
            if (s2 != null && !s2.isOver() && !next.isEmpty()) {
                double[][] feats = this.encodeAll(s2, next);
                double[] qOnline = forwardRows(this.getModel(), feats);
                int best = 0;
                for (int k = 1; k < qOnline.length; k++) {
                    if (qOnline[k] > qOnline[best]) best = k;
                }
                double[] qTarget = forwardRows(this.targetNetwork, new double[][] {feats[best]});
                target += discountFactor * qTarget[0];
            }
            y.set(i, 0, target);
        }
        return y;
    }

    public static void copyParams(Model from, Model to)
    {
        List<Parameter> src = from.getParameters();
        List<Parameter> dst = to.getParameters();
        try {
            for (int i = 0; i < src.size(); i++) {
                Matrix s = src.get(i).getValue();
                dst.get(i).getValue().copySlice(0, s.getShape().getNumRows(), 0, s.getShape().getNumCols(), s);
            }
        } catch (Exception e) {
            throw new RuntimeException("PolicyAgent: parameter copy failed", e);
        }
    }

    @Override
    public void afterGameEnds(BattleView view)
    {
        this.consecutiveSwitches = 0;
    }
}
