package src.pas.pokemon.train;


// SYSTEM IMPORTS
import net.sourceforge.argparse4j.inf.Namespace;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import edu.bu.pas.pokemon.core.Agent;
import edu.bu.pas.pokemon.core.Battle;
import edu.bu.pas.pokemon.core.Move;
import edu.bu.pas.pokemon.generators.BattleCreator;
import edu.bu.pas.pokemon.utils.Pair;


/**
 * Plays many greedy games of a PolicyAgent (loaded from a params file) against fixed opponents
 * and reports win/loss/draw rates per opponent.
 *
 * usage: Evaluate <model file> <games per opponent> <threads> <seed> <minTeam> <maxTeam> <opponent classes...>
 * Set -Dagent=<class> to evaluate a different agent class.
 */
public class Evaluate
{
    public static final int MAX_TURNS = 1000;

    public static Namespace makeNamespace(String inFile)
    {
        Map<String, Object> m = new HashMap<>();
        m.put("inFile", inFile == null ? "" : inFile);
        return new Namespace(m);
    }

    /** The engine prints every move to stdout; mute it and return the real stream for our own output. */
    public static java.io.PrintStream silenceStdout()
    {
        java.io.PrintStream real = System.out;
        System.setOut(new java.io.PrintStream(new java.io.OutputStream() {
            @Override public void write(int b) {}
            @Override public void write(byte[] b, int off, int len) {}
        }));
        return real;
    }

    public static Agent newAgent(String className, Namespace ns) throws Exception
    {
        Agent a = (Agent) Class.forName(className).getConstructor().newInstance();
        a.initialize(ns);
        return a;
    }

    /** @return 0 if team1 won, 1 if team2 won, -1 for a draw or turn cap */
    public static int playGame(Battle battle) throws Exception
    {
        int turns = 0;
        while (!battle.isOver() && turns < MAX_TURNS) {
            battle.nextTurn();
            battle.applyPreTurnConditions();
            Pair<Move, Move> moves = battle.getMoves();
            battle.applyMoves(moves);
            battle.applyPostTurnConditions();
            turns++;
        }
        battle.getTeam1().getAgent().afterGameEnds(battle.getView());
        battle.getTeam2().getAgent().afterGameEnds(battle.getView());
        boolean t1Alive = battle.getTeam1().getNumAlivePokemon() > 0;
        boolean t2Alive = battle.getTeam2().getNumAlivePokemon() > 0;
        if (t1Alive && !t2Alive) return 0;
        if (!t1Alive && t2Alive) return 1;
        return -1;
    }

    public static void main(String[] args) throws Exception
    {
        final String modelFile = args[0];
        final int numGames = Integer.parseInt(args[1]);
        final int numThreads = Integer.parseInt(args[2]);
        final long seed = Long.parseLong(args[3]);
        final int minTeam = Integer.parseInt(args[4]);
        final int maxTeam = Integer.parseInt(args[5]);
        final String agentClass = System.getProperty("agent", "src.pas.pokemon.agents.PolicyAgent");

        final java.io.PrintStream out = silenceStdout();
        final Namespace ns = makeNamespace(modelFile);
        // daemon workers: if a game throws, main reports it and the JVM exits instead of hanging on idle workers
        ExecutorService pool = Executors.newFixedThreadPool(numThreads, r -> {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        });

        for (int o = 6; o < args.length; o++) {
            final String oppClass = args[o];
            final long t0 = System.currentTimeMillis();
            List<Future<Integer>> results = new ArrayList<>();
            // same per-game seeds and equal team sizes as DQNTrainer.evalAgainst, so numbers are comparable
            // seed 0 reproduces DQNTrainer's eval games; any other seed gives a fresh held-out set
            final long seedBase = 10_000_000L * seed + (oppClass.endsWith("RandomAgent") ? 1_000_000L
                : oppClass.endsWith("AggroAgent") ? 2_000_000L : 4_000_000L);
            final int chunk = (numGames + numThreads - 1) / numThreads;
            for (int c = 0; c < numThreads; c++) {
                final int start = c * chunk;
                final int end = Math.min(numGames, start + chunk);
                if (start >= end) break;
                results.add(pool.submit(() -> {
                    // one learner instance per thread so per-instance state is never shared across threads
                    Agent me = newAgent(agentClass, ns);
                    Agent opp = newAgent(oppClass, makeNamespace(""));
                    int wins = 0, draws = 0;
                    for (int g = start; g < end; g++) {
                        Random rng = new Random(seedBase + g);
                        int n = minTeam + rng.nextInt(maxTeam - minTeam + 1);
                        int outcome = playGame(Battles.makeRandomTeams(n, n, 4, rng, me, opp));
                        if (outcome == 0) wins++;
                        else if (outcome == -1) draws++;
                    }
                    return wins * 100000 + draws;
                }));
            }
            int wins = 0, draws = 0;
            for (Future<Integer> f : results) {
                int r = f.get();
                wins += r / 100000;
                draws += r % 100000;
            }
            double secs = (System.currentTimeMillis() - t0) / 1000.0;
            out.println(String.format("EVAL opp=%s games=%d win=%.3f draw=%.3f loss=%.3f secs=%.1f",
                oppClass.substring(oppClass.lastIndexOf('.') + 1), numGames,
                wins / (double) numGames, draws / (double) numGames,
                (numGames - wins - draws) / (double) numGames, secs));
        }
        pool.shutdown();
    }
}
