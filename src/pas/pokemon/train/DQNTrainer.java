package src.pas.pokemon.train;


// SYSTEM IMPORTS
import java.io.File;
import java.io.FileWriter;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

import edu.bu.pas.pokemon.core.Agent;
import edu.bu.pas.pokemon.core.Battle;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.SwitchMove;
import edu.bu.pas.pokemon.generators.BattleCreator;
import edu.bu.pas.pokemon.nn.Model;
import edu.bu.pas.pokemon.utils.Pair;

// JAVA PROJECT IMPORTS
import src.pas.pokemon.agents.PolicyAgent;
import src.pas.pokemon.rewards.CustomRewardFunction;
import src.pas.pokemon.senses.CustomSensorArray;


/**
 * Double DQN trainer with a self-play opponent pool.
 *
 * Differences from the course's ParallelTrain:
 *   - transitions are recorded as (state the action was chosen in, that action, resulting state);
 *     the framework's TrainingBattleDriver pairs each action with the *next* turn's transition
 *   - minibatch updates with a target network synced every targetSync gradient steps
 *     (the framework recomputes targets once per cycle and then trains many epochs on them)
 *   - opponents are a mix of RandomAgent, AggroAgent and frozen snapshots of the learner
 *   - team sizes are randomised so the agent sees 1v1 ... 6v6 endgames, as in the graded matchups
 *
 * usage: DQNTrainer key=value ...   (see Config for keys and defaults)
 */
public class DQNTrainer
{
    static final String RANDOM = "edu.bu.pas.pokemon.agents.RandomAgent";
    static final String AGGRO = "edu.bu.pas.pokemon.agents.AggroAgent";
    static final String DRAC = "edu.bu.pas.pokemon.agents.Drac5290Agent";

    /** An opponent for one game; lookahead opponents (Drac) are kept to small teams because they are slow. */
    static final class Opponent
    {
        final Agent agent;
        final boolean smallTeams;
        Opponent(Agent agent, boolean smallTeams) { this.agent = agent; this.smallTeams = smallTeams; }
    }

    // ------------------------------------------------------------------ config

    static final class Config
    {
        final Map<String, String> kv = new HashMap<>();

        Config(String[] args)
        {
            for (String a : args) {
                int eq = a.indexOf('=');
                if (eq <= 0) throw new IllegalArgumentException("expected key=value, got " + a);
                kv.put(a.substring(0, eq), a.substring(eq + 1));
            }
        }

        String s(String k, String d) { return kv.getOrDefault(k, d); }
        int i(String k, int d) { return Integer.parseInt(s(k, Integer.toString(d))); }
        long l(String k, long d) { return Long.parseLong(s(k, Long.toString(d))); }
        double d(String k, double d) { return Double.parseDouble(s(k, Double.toString(d))); }
    }

    // ------------------------------------------------------------------ replay buffer

    static final class Transition
    {
        final float[] x;        // features of (s, a)
        final float reward;
        final boolean done;
        final float[][] next;   // features of (s_{t+n}, a') for every legal a' (empty when done)
        final float discount;   // gamma^n for the bootstrap term (n can be shorter at the end of a game)

        Transition(float[] x, float reward, boolean done, float[][] next, float discount)
        {
            this.x = x; this.reward = reward; this.done = done; this.next = next; this.discount = discount;
        }
    }

    static final class Replay
    {
        final Transition[] data;
        int size = 0, head = 0;

        Replay(int cap) { data = new Transition[cap]; }

        synchronized void addAll(List<Transition> ts)
        {
            for (Transition t : ts) {
                data[head] = t;
                head = (head + 1) % data.length;
                size = Math.min(size + 1, data.length);
            }
        }

        Transition sample(Random rng) { return data[rng.nextInt(size)]; }
    }

    static float[] toFloat(double[] v)
    {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) f[i] = (float) v[i];
        return f;
    }

    static double[] toDouble(float[] v)
    {
        double[] d = new double[v.length];
        for (int i = 0; i < v.length; i++) d[i] = v[i];
        return d;
    }

    // ------------------------------------------------------------------ opponents

    static final class Snapshot
    {
        final int cycle;
        final FastMLP net;
        Snapshot(int cycle, FastMLP net) { this.cycle = cycle; this.net = net; }
    }

    // ------------------------------------------------------------------ state

    final Config cfg;
    final PrintStream out;
    final Random rng;
    final FastMLP online;
    final FastMLP target;
    final Model frameworkModel; // only used for saving params files
    final Replay replay;
    final List<Snapshot> pool = new ArrayList<>();
    final ExecutorService workers;
    final AtomicLong gamesPlayed = new AtomicLong();
    final AtomicLong envSteps = new AtomicLong();
    long gradSteps = 0;

    DQNTrainer(Config cfg, PrintStream out) throws Exception
    {
        this.cfg = cfg;
        this.out = out;
        long seed = cfg.l("seed", System.nanoTime());
        this.rng = new Random(seed);
        PolicyAgent proto = new PolicyAgent();
        this.frameworkModel = proto.initModel();
        this.online = new FastMLP(CustomSensorArray.NUM_FEATURES, PolicyAgent.HIDDEN1, PolicyAgent.HIDDEN2, rng);
        String init = cfg.s("init", "");
        if (!init.isEmpty()) {
            frameworkModel.load(init);
            online.fromFramework(frameworkModel);
            out.println("loaded init params from " + init);
        }
        this.target = online.copy();
        this.replay = new Replay(cfg.i("buffer", 150000));
        this.workers = Executors.newFixedThreadPool(cfg.i("threads", 16));
    }

    double epsilon()
    {
        double start = cfg.d("epsStart", 1.0), end = cfg.d("epsEnd", 0.05);
        double frac = Math.min(1.0, gamesPlayed.get() / (double) cfg.l("epsDecayGames", 60000));
        return end + (start - end) * (1.0 - frac);
    }

    // ------------------------------------------------------------------ playing

    /** Builds the opponent for one training game according to the mixture weights. */
    Opponent pickOpponent(Random r)
    {
        double pDrac = cfg.d("pDrac", 0.10), pRandom = cfg.d("pRandom", 0.15), pAggro = cfg.d("pAggro", 0.30);
        double u = r.nextDouble();
        if (u < pDrac) return new Opponent(newScripted(DRAC), true);
        u = (u - pDrac) / (1.0 - pDrac);
        if (pool.isEmpty()) {
            // before the first snapshot exists, split the self-play share between the scripted agents
            return new Opponent(u < pRandom / (pRandom + pAggro) ? newScripted(RANDOM) : newScripted(AGGRO), false);
        }
        double rest = 1.0 - pDrac;
        if (u < pRandom / rest) return new Opponent(newScripted(RANDOM), false);
        if (u < (pRandom + pAggro) / rest) return new Opponent(newScripted(AGGRO), false);
        // self-play: favour recent snapshots but keep older ones in the mix
        int n = pool.size();
        int idx = r.nextDouble() < 0.5 ? n - 1 : r.nextInt(n);
        return new Opponent(snapshotAgent(pool.get(idx).net, cfg.d("oppEps", 0.05)), false);
    }

    static Agent newScripted(String cls)
    {
        try {
            Agent a = (Agent) Class.forName(cls).getConstructor().newInstance();
            a.initialize(Evaluate.makeNamespace(""));
            return a;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    PolicyAgent snapshotAgent(FastMLP net, double eps)
    {
        PolicyAgent a = new PolicyAgent();
        a.useModel(frameworkModel, net);
        a.setExplorationEpsilon(eps);
        return a;
    }

    int teamSize(Random r, int hi)
    {
        int lo = cfg.i("minTeam", 1);
        // weight towards bigger teams: a full 6v6 still contains the small endgames
        int a = lo + r.nextInt(hi - lo + 1), b = lo + r.nextInt(hi - lo + 1);
        return Math.max(a, b);
    }

    /** One learner decision before n-step aggregation. */
    static final class Step
    {
        final float[] x; final float reward; final boolean done; final float[][] next;
        Step(float[] x, float reward, boolean done, float[][] next) { this.x = x; this.reward = reward; this.done = done; this.next = next; }
    }

    /**
     * Plays one training game and returns the learner's n-step transitions. The learner is team 1 or
     * team 2 at random (the engine resolves simultaneous faint replacements team 1 first), and each
     * game gets fresh agents because Agent.registerTeamIdx only takes the first index it sees.
     */
    List<Transition> playTrainingGame(Opponent opponent, double eps, Random r) throws Exception
    {
        PolicyAgent learner = new PolicyAgent();
        learner.useModel(frameworkModel, online);
        learner.setExplorationEpsilon(eps);
        Agent opp = opponent.agent;
        int me = r.nextDouble() < cfg.d("pTeam2", 0.5) ? 1 : 0;
        int hi = opponent.smallTeams ? cfg.i("dracTrainMaxTeam", 2) : cfg.i("maxTeam", 6);
        int nMine = teamSize(r, hi);
        int nOpp = r.nextDouble() < 0.5 ? nMine : teamSize(r, hi);
        Battle battle = me == 0
            ? Battles.makeRandomTeams(nMine, nOpp, 4, r, learner, opp)
            : Battles.makeRandomTeams(nOpp, nMine, 4, r, opp, learner);
        CustomSensorArray sensors = (CustomSensorArray) learner.getSensorArray();
        List<Step> steps = new ArrayList<>(64);
        int turns = 0;
        int switchRun = 0; // mirrors PolicyAgent's consecutive-switch guard
        while (!battle.isOver() && turns < Evaluate.MAX_TURNS) {
            battle.nextTurn();
            battle.applyPreTurnConditions();
            BattleView s = battle.getView();
            boolean forced = s.getTeamView(me).getActivePokemonView().getActiveMoveView() != null;
            Pair<Move, Move> moves = battle.getMoves();
            battle.applyMoves(moves);
            battle.applyPostTurnConditions();
            BattleView s2 = battle.getView();
            turns++;
            Move mine = me == 0 ? moves.getFirst() : moves.getSecond();
            if (mine == null) {
                continue;
            }
            MoveView a = mine.getView();
            if (!forced) {
                switchRun = (a instanceof SwitchMove.SwitchMoveView) ? switchRun + 1 : 0;
            }
            boolean done = s2.isOver();
            float rew = (float) CustomRewardFunction.reward(s, a, s2, me);
            float[][] next;
            if (done) {
                next = new float[0][];
            } else {
                List<MoveView> legal = guarded(PolicyAgent.legalMovesFor(s2, me), switchRun);
                next = new float[legal.size()][];
                for (int k = 0; k < legal.size(); k++) {
                    next[k] = toFloat(sensors.encode(s2, legal.get(k), me));
                }
            }
            steps.add(new Step(toFloat(sensors.encode(s, a, me)), rew, done, next));
        }
        learner.afterGameEnds(battle.getView());
        opp.afterGameEnds(battle.getView());
        envSteps.addAndGet(steps.size());
        return nStep(steps);
    }

    /** Same filtering PolicyAgent.getMove applies after too many switches in a row. */
    static List<MoveView> guarded(List<MoveView> legal, int switchRun)
    {
        if (switchRun < PolicyAgent.MAX_CONSECUTIVE_SWITCHES) {
            return legal;
        }
        List<MoveView> attacks = new ArrayList<>();
        for (MoveView m : legal) {
            if (!(m instanceof SwitchMove.SwitchMoveView)) attacks.add(m);
        }
        return attacks.isEmpty() ? legal : attacks;
    }

    /** n-step returns: R = r_t + g r_{t+1} + ... and bootstrap from the state n steps later (or stop at the end). */
    List<Transition> nStep(List<Step> steps)
    {
        int n = cfg.i("nStep", 3);
        double gamma = cfg.d("gamma", 0.97);
        List<Transition> ts = new ArrayList<>(steps.size());
        for (int t = 0; t < steps.size(); t++) {
            double ret = 0.0, disc = 1.0;
            int k = t;
            while (true) {
                Step st = steps.get(k);
                ret += disc * st.reward;
                disc *= gamma;
                if (st.done || k == steps.size() - 1 || k - t + 1 == n) break;
                k++;
            }
            Step last = steps.get(k);
            ts.add(new Transition(steps.get(t).x, (float) ret, last.done, last.next, (float) disc));
        }
        return ts;
    }

    /** Workers pull games from a shared counter so a few slow games (Drac lookahead) don't idle the other threads. */
    void collect(int numGames, double eps) throws Exception
    {
        int threads = cfg.i("threads", 16);
        java.util.concurrent.atomic.AtomicInteger nextGame = new java.util.concurrent.atomic.AtomicInteger();
        List<Future<?>> fs = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
            final long seed = rng.nextLong();
            fs.add(workers.submit(() -> {
                Random r = new Random(seed);
                List<Transition> buf = new ArrayList<>();
                while (nextGame.getAndIncrement() < numGames) {
                    buf.addAll(playTrainingGame(pickOpponent(r), eps, r));
                    gamesPlayed.incrementAndGet();
                }
                replay.addAll(buf);
                return null;
            }));
        }
        for (Future<?> f : fs) f.get();
    }

    // ------------------------------------------------------------------ learning

    /** @return mean squared TD error over the updates */
    double learn(int numUpdates, double lr)
    {
        int B = cfg.i("batch", 256);
        double huber = cfg.d("huber", 1.0);
        int sync = cfg.i("targetSync", 500);
        double lossSum = 0.0;
        for (int u = 0; u < numUpdates; u++) {
            Transition[] batch = new Transition[B];
            int totalNext = 0;
            for (int i = 0; i < B; i++) {
                batch[i] = replay.sample(rng);
                totalNext += batch[i].next.length;
            }
            // Double DQN: pick a' with the online net, evaluate it with the target net
            double[][] nextX = new double[totalNext][];
            int k = 0;
            for (Transition t : batch) {
                for (float[] f : t.next) nextX[k++] = toDouble(f);
            }
            double[] qOnline = online.scoreParallel(nextX);
            double[][] chosen = new double[B][];
            k = 0;
            for (int i = 0; i < B; i++) {
                Transition t = batch[i];
                if (t.next.length == 0) continue;
                int best = k;
                for (int j = k; j < k + t.next.length; j++) {
                    if (qOnline[j] > qOnline[best]) best = j;
                }
                chosen[i] = nextX[best];
                k += t.next.length;
            }
            double[] y = new double[B];
            double[][] X = new double[B][];
            for (int i = 0; i < B; i++) {
                Transition t = batch[i];
                X[i] = toDouble(t.x);
                y[i] = t.reward;
                if (!t.done && chosen[i] != null) {
                    y[i] += t.discount * target.forwardOne(chosen[i]);
                }
            }
            lossSum += online.trainStep(X, y, lr, huber, cfg.d("gradClip", 10.0));
            gradSteps++;
            if (gradSteps % sync == 0) {
                target.copyFrom(online);
            }
        }
        return numUpdates == 0 ? 0.0 : lossSum / numUpdates;
    }

    // ------------------------------------------------------------------ evaluation

    /**
     * Greedy win rate on a fixed set of seeds. Each seed is played twice with the teams swapped
     * (learner gets team A, then team B), which cancels most of the team-draw luck.
     */
    double evalAgainst(final java.util.function.Supplier<Agent> oppFactory, int pairs, long seedBase,
                       final int minTeam, final int maxTeam) throws Exception
    {
        int threads = cfg.i("threads", 16);
        java.util.concurrent.atomic.AtomicInteger nextPair = new java.util.concurrent.atomic.AtomicInteger();
        List<Future<Integer>> fs = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
            fs.add(workers.submit(() -> {
                int wins = 0;
                int g;
                while ((g = nextPair.getAndIncrement()) < pairs) {
                    for (int side = 0; side < 2; side++) {
                        Random r = new Random(seedBase + g);
                        int n = minTeam + r.nextInt(maxTeam - minTeam + 1);
                        PolicyAgent me = snapshotAgent(online, 0.0);
                        Agent opp = oppFactory.get();
                        Battle b = side == 0
                            ? Battles.makeRandomTeams(n, n, 4, r, me, opp)
                            : Battles.makeRandomTeams(n, n, 4, r, opp, me);
                        if (Evaluate.playGame(b) == side) wins++;
                    }
                }
                return wins;
            }));
        }
        int wins = 0;
        for (Future<Integer> f : fs) wins += f.get();
        return wins / (2.0 * pairs);
    }

    // ------------------------------------------------------------------ main loop

    void run() throws Exception
    {
        String outDir = cfg.s("out", "runs/run");
        new File(outDir).mkdirs();
        PrintWriter csv = new PrintWriter(new FileWriter(new File(outDir, "metrics.csv"), true), true);
        csv.println("cycle,games,env_steps,grad_steps,epsilon,lr,buffer,loss,win_random,win_aggro,win_snapshot,win_drac,secs");

        int cycles = cfg.i("cycles", 1000000);
        double hours = cfg.d("hours", 0.0);           // 0 = no time limit
        int gamesPerCycle = cfg.i("gamesPerCycle", 256);
        double replayRatio = cfg.d("replayRatio", 4.0);
        int batch = cfg.i("batch", 256);
        int warmup = cfg.i("warmup", 20000);
        int evalEvery = cfg.i("evalEvery", 10);
        int evalPairs = cfg.i("evalPairs", 500);
        int dracEvalEvery = cfg.i("dracEvalEvery", 50);
        int dracEvalPairs = cfg.i("dracEvalPairs", 100);
        int checkpointEvery = cfg.i("checkpointEvery", 100);
        int snapshotEvery = cfg.i("snapshotEvery", 25);
        int poolSize = cfg.i("poolSize", 12);
        double lrStart = cfg.d("lr", 2e-4), lrEnd = cfg.d("lrEnd", lrStart);
        int minTeam = cfg.i("minTeam", 1), maxTeam = cfg.i("maxTeam", 6);
        double bestScore = -1.0;
        double lastDrac = Double.NaN;
        long t0 = System.currentTimeMillis();

        out.println("features=" + CustomSensorArray.NUM_FEATURES + " hidden=" + PolicyAgent.HIDDEN1 + "," + PolicyAgent.HIDDEN2
            + " config=" + cfg.kv);
        for (int c = 1; c <= cycles; c++) {
            double elapsedH = (System.currentTimeMillis() - t0) / 3.6e6;
            if (hours > 0 && elapsedH >= hours) {
                out.println("time limit reached after " + (c - 1) + " cycles");
                break;
            }
            double progress = hours > 0 ? elapsedH / hours : (c - 1) / (double) cycles;
            double lr = lrStart + (lrEnd - lrStart) * Math.min(1.0, progress);
            double eps = epsilon();
            long stepsBefore = envSteps.get();
            collect(gamesPerCycle, eps);
            long newSteps = envSteps.get() - stepsBefore;

            double loss = 0.0;
            if (replay.size >= warmup) {
                int updates = (int) Math.max(1, Math.round(newSteps * replayRatio / batch));
                loss = learn(updates, lr);
            }

            if (c % snapshotEvery == 0) {
                pool.add(new Snapshot(c, online.copy()));
                if (pool.size() > poolSize) pool.remove(0);
            }

            String evalPart = "";
            if (c % evalEvery == 0) {
                double wr = evalAgainst(() -> newScripted(RANDOM), evalPairs, 1_000_000L, minTeam, maxTeam);
                double wa = evalAgainst(() -> newScripted(AGGRO), evalPairs, 2_000_000L, minTeam, maxTeam);
                final FastMLP oldest = pool.isEmpty() ? null : pool.get(0).net;
                double ws = oldest == null ? Double.NaN
                    : evalAgainst(() -> snapshotAgent(oldest, 0.0), evalPairs / 4, 3_000_000L, minTeam, maxTeam);
                if (c % dracEvalEvery == 0) {
                    lastDrac = evalAgainst(() -> newScripted(DRAC), dracEvalPairs, 4_000_000L, 1, cfg.i("dracMaxTeam", 3));
                }
                double secs = (System.currentTimeMillis() - t0) / 1000.0;
                evalPart = String.format(" | eval win: random=%.3f aggro=%.3f oldest_snapshot=%.3f drac(1-%d)=%.3f",
                    wr, wa, ws, cfg.i("dracMaxTeam", 3), lastDrac);
                csv.println(String.format("%d,%d,%d,%d,%.4f,%.2e,%d,%.5f,%.4f,%.4f,%.4f,%.4f,%.0f",
                    c, gamesPlayed.get(), envSteps.get(), gradSteps, eps, lr, replay.size, loss, wr, wa, ws, lastDrac, secs));

                online.toFramework(frameworkModel);
                if (c % checkpointEvery == 0) {
                    frameworkModel.save(new File(outDir, "cycle" + c + ".model").getPath());
                }
                double score = 0.5 * (wr + wa);
                if (score > bestScore) {
                    bestScore = score;
                    frameworkModel.save(new File(outDir, "best.model").getPath());
                    evalPart += " (new best)";
                }
            }
            // MetronomeCallback (unpatched jar) builds a fresh CoreRegistry whose unclosed jar streams hold
            // native zlib memory until their finalizers run; force that periodically
            if (c % 10 == 0) {
                System.gc();
                System.runFinalization();
            }
            Runtime rt = Runtime.getRuntime();
            out.println(String.format("cycle=%d games=%d steps=%d grad=%d eps=%.3f lr=%.1e buf=%d loss=%.4f heap=%dM rss=%dM t=%.0fs%s",
                c, gamesPlayed.get(), envSteps.get(), gradSteps, eps, lr, replay.size, loss,
                (rt.totalMemory() - rt.freeMemory()) >> 20, rssMb(),
                (System.currentTimeMillis() - t0) / 1000.0, evalPart));
        }
        online.toFramework(frameworkModel);
        frameworkModel.save(new File(outDir, "final.model").getPath());
        csv.close();
        workers.shutdown();
    }

    /** Resident set size of this process, in MB (Linux only; -1 elsewhere). */
    static long rssMb()
    {
        try {
            for (String line : java.nio.file.Files.readAllLines(java.nio.file.Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmRSS:")) {
                    return Long.parseLong(line.replaceAll("[^0-9]", "")) >> 10;
                }
            }
        } catch (Exception e) {
            // not on Linux
        }
        return -1;
    }

    public static void main(String[] args) throws Exception
    {
        PrintStream out = Evaluate.silenceStdout();
        new DQNTrainer(new Config(args), out).run();
    }
}
