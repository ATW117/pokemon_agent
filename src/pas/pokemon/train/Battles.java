package src.pas.pokemon.train;


// SYSTEM IMPORTS
import java.util.List;
import java.util.Random;

import edu.bu.pas.pokemon.core.Agent;
import edu.bu.pas.pokemon.core.Battle;
import edu.bu.pas.pokemon.core.CoreRegistry;
import edu.bu.pas.pokemon.core.Move;
import edu.bu.pas.pokemon.core.Pokemon;
import edu.bu.pas.pokemon.core.Team;
import edu.bu.pas.pokemon.generators.BattleCreator;


/**
 * Same teams as BattleCreator.makeRandomTeams, but reuses one CoreRegistry per thread.
 *
 * BattleCreator builds a new CoreRegistry for every battle, and CoreRegistry.readCSV never closes
 * its jar resource stream on success. Each game therefore leaks native zlib buffers until a full GC
 * happens to finalize them (observed: ~700MB of RSS growth per training cycle). The registry hands
 * out copies (getMove -> Move.copy(), spawnPokemon -> new Pokemon), so sharing it within a thread is safe.
 */
public final class Battles
{
    private static final ThreadLocal<CoreRegistry> REGISTRY = ThreadLocal.withInitial(() -> {
        try {
            return new CoreRegistry();
        } catch (Exception e) {
            throw new RuntimeException("Battles: could not build CoreRegistry", e);
        }
    });

    private Battles() {}

    public static Battle makeRandomTeams(int n1, int n2, int numMoves, Random rng, Agent agent1, Agent agent2)
    {
        CoreRegistry reg = REGISTRY.get();
        List<Integer> dex1 = BattleCreator.getRandomDexIdxs(n1, rng, reg);
        List<Integer> dex2 = BattleCreator.getRandomDexIdxs(n2, rng, reg);
        Team t1 = makeTeam(dex1, numMoves, rng, reg, agent1);
        Team t2 = makeTeam(dex2, numMoves, rng, reg, agent2);
        return new Battle(t1, t2);
    }

    private static Team makeTeam(List<Integer> dex, int numMoves, Random rng, CoreRegistry reg, Agent agent)
    {
        Pokemon[] mons = new Pokemon[dex.size()];
        for (int j = 0; j < mons.length; j++) {
            int d = dex.get(j);
            Move[] moves = BattleCreator.chooseRandomUniqueMoves(d, numMoves, rng, reg);
            mons[j] = reg.spawnPokemon(d, 100, new int[] {15, 15, 15, 15, 15, 15, 1, 1},
                new int[] {65535, 65535, 65535, 65535, 65535, 65535, 1, 1}, moves);
        }
        return new Team(mons, agent);
    }
}
