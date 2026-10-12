package src.pas.pokemon.senses;


// SYSTEM IMPORTS
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntSupplier;

// JAVA PROJECT IMPORTS
import edu.bu.pas.pokemon.agents.senses.SensorArray;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.CoreRegistry;
import edu.bu.pas.pokemon.core.callbacks.Callback;
import edu.bu.pas.pokemon.core.callbacks.MultiCallbackCallback;
import edu.bu.pas.pokemon.core.callbacks.RepeatedWithinThisTurnCallback;
import edu.bu.pas.pokemon.core.callbacks.StochasticIfMoveHitsCallback;
import edu.bu.pas.pokemon.core.Move;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.Pokemon.PokemonView;
import edu.bu.pas.pokemon.core.SwitchMove.SwitchMoveView;
import edu.bu.pas.pokemon.core.Team.TeamView;
import edu.bu.pas.pokemon.core.enums.NonVolatileStatus;
import edu.bu.pas.pokemon.core.enums.Stat;
import edu.bu.pas.pokemon.core.enums.Height;
import edu.bu.pas.pokemon.core.enums.Type;
import edu.bu.pas.pokemon.core.enums.Flag;
import edu.bu.pas.pokemon.linalg.Matrix;


/**
 * Encodes a (state, action) pair from the point of view of the team that owns this sensor array.
 *
 * Layout: [state block][action block]. The action block has a shared header plus an attack section
 * (zeros for switches) and a switch-target section (zeros for attacks), so a SwitchMove to slot 2
 * and a SwitchMove to slot 4 produce different inputs.
 */
public class CustomSensorArray
    extends SensorArray
{
    private static final Type[] ALL_TYPES = Type.values();
    private static final Height[] ALL_HEIGHTS = Height.values();
    private static final int NT = ALL_TYPES.length;
    private static final int NH = ALL_HEIGHTS.length;
    private static final int NS = 7; // non-volatile statuses we one-hot

    // Type.getEffectivenessModifier is static synchronized, which serializes worker threads; cache it once
    private static final double[][] TYPE_CHART = buildTypeChart();

    // move identity: many status moves are otherwise indistinguishable (Swords Dance vs Leer vs Tail Whip)
    private static final Map<String, Integer> MOVE_INDEX = new HashMap<>();
    private static final Map<String, Double> EXPECTED_HITS = new HashMap<>();
    private static final int NM;
    static {
        List<String> names = new ArrayList<>();
        try {
            CoreRegistry reg = new CoreRegistry();
            names.addAll(reg.getAllMoveNames());
            Collections.sort(names);
            for (String n : names) {
                double hits = expectedHits(reg.getMove(n).getCallbacks());
                if (hits != 1.0) EXPECTED_HITS.put(n, hits);
            }
        } catch (Exception e) {
            throw new RuntimeException("CustomSensorArray: could not read the move database", e);
        }
        for (int i = 0; i < names.size(); i++) MOVE_INDEX.put(names.get(i), i);
        NM = names.size();
    }

    // ---- feature counts (kept next to the writers below; checked at runtime) ----
    private static final int ACTIVE_FEATS = 1 + NS + 2 + 7 + 6 + NH + NT;          // per active pokemon
    private static final int BENCH_FEATS = 6 * (3 + NS);                         // per team
    private static final int STATE_FEATS = 2 * ACTIVE_FEATS
        + 4                // speed comparison
        + 2 * BENCH_FEATS
        + 6                // team counts
        + 4                // screens
        + 4;               // current matchup summary
    private static final int ACTION_HEADER = 2;                                  // isAttack, isSwitch
    private static final int ATTACK_FEATS = 3 + 4 + 3 + 2 + 1 + NT + 5 + 2 + NM;
    private static final int SWITCH_FEATS = 1 + NS + NT + 6;
    public static final int NUM_FEATURES = STATE_FEATS + ACTION_HEADER + ATTACK_FEATS + SWITCH_FEATS;

    private final IntSupplier myTeamIdx;

    public CustomSensorArray()
    {
        this(() -> 0);
    }

    /** @param myTeamIdx supplies the owning agent's team index (0 or 1), read on every call */
    public CustomSensorArray(IntSupplier myTeamIdx)
    {
        this.myTeamIdx = myTeamIdx;
    }

    public Matrix getSensorValues(final BattleView state, final MoveView action)
    {
        double[] f = encode(state, action, this.perspective());
        final Matrix out = Matrix.zeros(1, f.length);
        for (int i = 0; i < f.length; i++) {
            out.set(0, i, f[i]);
        }
        return out;
    }

    private int perspective()
    {
        int idx = this.myTeamIdx.getAsInt();
        return idx == 1 ? 1 : 0;
    }

    /** Fast path used by the trainer: same features as getSensorValues without the Matrix wrapper. */
    public double[] encode(final BattleView state, final MoveView action, final int teamIdx)
    {
        final TeamView myTeam = state.getTeamView(teamIdx);
        final TeamView oppTeam = state.getTeamView(1 - teamIdx);
        final PokemonView me = myTeam.getActivePokemonView();
        final PokemonView opp = oppTeam.getActivePokemonView();

        final Writer w = new Writer(NUM_FEATURES);

        // ---------------- state ----------------
        writeActive(w, me);
        writeActive(w, opp);

        final double mySpd = effectiveSpeed(me);
        final double oppSpd = effectiveSpeed(opp);
        w.add(mySpd > oppSpd ? 1.0 : 0.0);
        w.add(mySpd == oppSpd ? 1.0 : 0.0);
        w.add(clamp((mySpd - oppSpd) / Math.max(1.0, Math.max(mySpd, oppSpd)), -1.0, 1.0));
        w.add(clamp(mySpd / 400.0, 0.0, 1.5));

        writeBench(w, myTeam);
        writeBench(w, oppTeam);

        final int myAlive = countAlive(myTeam);
        final int oppAlive = countAlive(oppTeam);
        w.add(myTeam.size() / 6.0);
        w.add(oppTeam.size() / 6.0);
        w.add(myAlive / 6.0);
        w.add(oppAlive / 6.0);
        w.add(teamHpFrac(myTeam));
        w.add(teamHpFrac(oppTeam));

        w.add(normCounter(myTeam.getNumReflectTurnsRemaining(), 5));
        w.add(normCounter(myTeam.getNumLightScreenTurnsRemaining(), 5));
        w.add(normCounter(oppTeam.getNumReflectTurnsRemaining(), 5));
        w.add(normCounter(oppTeam.getNumLightScreenTurnsRemaining(), 5));

        // how the current active matchup looks in both directions
        w.add(bestDamageFrac(me, opp, oppTeam));
        w.add(bestDamageFrac(opp, me, myTeam));
        w.add(log2Eff(bestEffectiveness(me, opp)));
        w.add(log2Eff(bestEffectiveness(opp, me)));

        // ---------------- action ----------------
        final boolean isSwitch = action instanceof SwitchMoveView;
        w.add(isSwitch || action == null ? 0.0 : 1.0);
        w.add(isSwitch ? 1.0 : 0.0);

        if (action != null && !isSwitch) {
            writeAttack(w, action, me, opp, oppTeam);
        } else {
            w.skip(ATTACK_FEATS);
        }

        if (isSwitch) {
            int idx = ((SwitchMoveView) action).getNewActiveIdx();
            writeSwitchTarget(w, myTeam.getPokemonView(idx), opp, oppTeam, myTeam, oppSpd);
        } else {
            w.skip(SWITCH_FEATS);
        }

        if (w.pos != NUM_FEATURES) {
            throw new IllegalStateException("CustomSensorArray wrote " + w.pos + " features, expected " + NUM_FEATURES);
        }
        return w.data;
    }

    // ------------------------------------------------------------------ blocks

    private static void writeActive(Writer w, PokemonView p)
    {
        w.add(hpFrac(p));
        writeStatusOneHot(w, p.getNonVolatileStatus());
        w.add(normCounter(p.getNonVolatileStatusCounter(NonVolatileStatus.SLEEP), 7));
        w.add(normCounter(p.getNonVolatileStatusCounter(NonVolatileStatus.FREEZE), 7));
        // stat stages live in [-6, 6]
        w.add(p.getStatMultiplier(Stat.ATK) / 6.0);
        w.add(p.getStatMultiplier(Stat.DEF) / 6.0);
        w.add(p.getStatMultiplier(Stat.SPD) / 6.0);
        w.add(p.getStatMultiplier(Stat.SPATK) / 6.0);
        w.add(p.getStatMultiplier(Stat.SPDEF) / 6.0);
        w.add(p.getStatMultiplier(Stat.ACC) / 6.0);
        w.add(p.getStatMultiplier(Stat.EVASIVE) / 6.0);
        w.add(p.getFlag(Flag.CONFUSED) ? 1.0 : 0.0);
        w.add(p.getFlag(Flag.TRAPPED) ? 1.0 : 0.0);
        w.add(p.getFlag(Flag.SEEDED) ? 1.0 : 0.0);
        w.add(p.getFlag(Flag.FOCUS_ENERGY) ? 1.0 : 0.0);
        w.add(p.getSubstitute() != null ? 1.0 : 0.0);
        w.add(p.getActiveMoveView() != null ? 1.0 : 0.0); // locked into a multi-turn move
        Height h = p.getHeight();
        for (Height val : ALL_HEIGHTS) {
            w.add(h == val ? 1.0 : 0.0);
        }
        writeTypes(w, p);
    }

    private static void writeBench(Writer w, TeamView team)
    {
        int size = team.size();
        int activeIdx = team.getActivePokemonIdx();
        for (int i = 0; i < 6; i++) {
            if (i >= size || team.getPokemonView(i) == null) {
                w.skip(3 + NS);
                continue;
            }
            PokemonView p = team.getPokemonView(i);
            w.add(i == activeIdx ? 1.0 : 0.0);
            w.add(p.hasFainted() ? 1.0 : 0.0);
            w.add(hpFrac(p));
            writeStatusOneHot(w, p.getNonVolatileStatus());
        }
    }

    private static void writeAttack(Writer w, MoveView a, PokemonView me, PokemonView opp, TeamView oppTeam)
    {
        Move.Category cat = a.getCategory();
        w.add(cat == Move.Category.PHYSICAL ? 1.0 : 0.0);
        w.add(cat == Move.Category.SPECIAL ? 1.0 : 0.0);
        w.add(cat == Move.Category.STATUS ? 1.0 : 0.0);

        Integer pow = a.getPower();
        Integer acc = a.getAccuracy();
        Integer pp = a.getPP();
        w.add(pow == null ? 0.0 : clamp(pow / 150.0, 0.0, 1.0));
        w.add(pow == null ? 1.0 : 0.0); // fixed/variable damage or no damage
        w.add(acc == null ? 1.0 : clamp(acc / 100.0, 0.0, 1.0));
        w.add(pp == null ? 1.0 : clamp(pp / 40.0, 0.0, 1.0));

        int pr = a.getPriority();
        w.add(pr < 0 ? 1.0 : 0.0);
        w.add(pr == 0 ? 1.0 : 0.0);
        w.add(pr > 0 ? 1.0 : 0.0);

        w.add(clamp(a.getCriticalHitRatio() / 8.0, 0.0, 1.0));
        boolean canHit = a.getCanHitHeights() != null && a.getCanHitHeights().contains(opp.getHeight());
        w.add(canHit ? 1.0 : 0.0);

        boolean stab = a.getType() != null && (a.getType() == me.getCurrentType1() || a.getType() == me.getCurrentType2());
        w.add(stab ? 1.0 : 0.0);

        Type mt = a.getType();
        for (Type t : ALL_TYPES) {
            w.add(mt == t ? 1.0 : 0.0);
        }

        double eff = effectiveness(mt, opp);
        double dmg = expectedDamageFrac(a, me, opp, oppTeam);
        w.add(log2Eff(eff));
        w.add(eff == 0.0 ? 1.0 : 0.0);
        w.add(eff > 1.0 ? 1.0 : 0.0);
        w.add(clamp(dmg, 0.0, 1.5));
        w.add(dmg >= 1.0 ? 1.0 : 0.0); // likely KO if it lands
        String name = a.getName();
        w.add(FIXED_DAMAGE_MOVES.contains(name) ? 1.0 : 0.0);
        w.add(OHKO_MOVES.contains(name) ? 1.0 : 0.0);
        Integer mi = MOVE_INDEX.get(name);
        for (int i = 0; i < NM; i++) {
            w.add(mi != null && mi == i ? 1.0 : 0.0);
        }
    }

    private static void writeSwitchTarget(Writer w, PokemonView target, PokemonView opp, TeamView oppTeam,
                                          TeamView myTeam, double oppSpd)
    {
        w.add(hpFrac(target));
        writeStatusOneHot(w, target.getNonVolatileStatus());
        writeTypes(w, target);
        w.add(log2Eff(bestEffectiveness(target, opp)));
        w.add(log2Eff(bestEffectiveness(opp, target)));
        w.add(clamp(bestDamageFrac(target, opp, oppTeam), 0.0, 1.5));
        // incoming damage measured against the target's *current* HP so a weakened switch-in looks risky
        w.add(clamp(bestDamageFrac(opp, target, myTeam), 0.0, 1.5));
        double tSpd = effectiveSpeed(target);
        w.add(tSpd > oppSpd ? 1.0 : 0.0);
        w.add(clamp((tSpd - oppSpd) / Math.max(1.0, Math.max(tSpd, oppSpd)), -1.0, 1.0));
    }

    // ------------------------------------------------------------------ helpers

    private static final java.util.Set<String> FIXED_DAMAGE_MOVES = new java.util.HashSet<>(java.util.Arrays.asList(
        "Seismic Toss", "Night Shade", "Dragon Rage", "Sonic Boom", "Super Fang", "Psywave"));
    private static final java.util.Set<String> OHKO_MOVES = new java.util.HashSet<>(java.util.Arrays.asList(
        "Fissure", "Guillotine", "Horn Drill"));

    /** Mean number of hits, from the move's RepeatedWithinThisTurnCallback (uniform over [min, max] in the engine). */
    private static double expectedHits(List<Callback> callbacks)
    {
        double hits = 1.0;
        if (callbacks == null) return hits;
        for (Callback c : callbacks) {
            hits = Math.max(hits, expectedHits(c));
        }
        return hits;
    }

    private static double expectedHits(Callback c)
    {
        if (c instanceof RepeatedWithinThisTurnCallback) {
            RepeatedWithinThisTurnCallback r = (RepeatedWithinThisTurnCallback) c;
            return (r.getMinReps() + r.getMaxReps()) / 2.0;
        }
        if (c instanceof MultiCallbackCallback) {
            return expectedHits(((MultiCallbackCallback) c).getCallbacks());
        }
        if (c instanceof StochasticIfMoveHitsCallback) {
            return expectedHits(((StochasticIfMoveHitsCallback) c).getHitCallback());
        }
        return 1.0;
    }

    /** Engine accuracy/evasion stage multiplier (Gen-1 style 3/(3+n) table). */
    private static double stageMultiplier(int stage)
    {
        return stage >= 0 ? (3.0 + stage) / 3.0 : 3.0 / (3.0 - stage);
    }

    /**
     * Expected damage of one use of the move as a fraction of the defender's current HP, mirroring
     * core/DamageEquation + DoDamageCallback: mean random roll, crit chance, STAB, type, screens (which
     * the engine applies twice, i.e. 1/4 damage), Explosion's halved defense, hit chance including
     * accuracy/evasion stages, multi-hit counts, fixed-damage and one-hit-KO moves.
     */
    public static double expectedDamageFrac(MoveView a, PokemonView atk, PokemonView def, TeamView defTeam)
    {
        if (a == null || a.getCategory() == Move.Category.STATUS) {
            return 0.0;
        }
        int defHp = def.getCurrentStat(Stat.HP);
        if (defHp <= 0) {
            return 0.0;
        }
        if (a.getCanHitHeights() != null && !a.getCanHitHeights().contains(def.getHeight())) {
            return 0.0; // target is underground / in the air
        }
        double eff = effectiveness(a.getType(), def);
        double acc = a.getAccuracy() == null ? 1.0 : clamp(a.getAccuracy() / 100.0, 0.0, 1.0);
        if (a.getAccuracy() != null) {
            acc = clamp(acc * stageMultiplier(atk.getStatMultiplier(Stat.ACC) - def.getStatMultiplier(Stat.EVASIVE)), 0.0, 1.0);
        }
        String name = a.getName();
        if (FIXED_DAMAGE_MOVES.contains(name)) {
            if (eff == 0.0) return 0.0; // immunity still applies (e.g. Night Shade vs Normal)
            double fixed;
            switch (name) {
                case "Dragon Rage": fixed = 40; break;
                case "Sonic Boom": fixed = 20; break;
                case "Super Fang": fixed = Math.max(1, defHp / 2); break;
                case "Psywave": fixed = 0.75 * atk.getLevel(); break;
                default: fixed = atk.getLevel(); break; // Seismic Toss, Night Shade
            }
            return acc * fixed / defHp;
        }
        if (OHKO_MOVES.contains(name)) {
            return eff == 0.0 ? 0.0 : acc; // KO if it hits
        }
        if (a.getPower() == null || a.getPower() <= 0) {
            return 0.0; // Counter, Bide, ... depend on history
        }
        boolean phys = a.getCategory() == Move.Category.PHYSICAL;
        double A = Math.max(1, atk.getCurrentStat(phys ? Stat.ATK : Stat.SPATK));
        double D = Math.max(1, def.getCurrentStat(phys ? Stat.DEF : Stat.SPDEF));
        if ("Explosion".equals(name) || "Self-Destruct".equals(name)) {
            D = Math.max(1, D / 2);
        }
        double screen = 1.0;
        if (defTeam != null && ((phys && defTeam.getNumReflectTurnsRemaining() > 0)
                || (!phys && defTeam.getNumLightScreenTurnsRemaining() > 0))) {
            screen = 0.25;
        }
        double base = (2.0 * atk.getLevel() / 5.0 + 2.0) * a.getPower() * (A / D) / 50.0 + 2.0;
        if (a.getType() == atk.getCurrentType1() || a.getType() == atk.getCurrentType2()) {
            base *= 1.5;
        }
        // crit: the engine keys the threshold off the defender's base speed; high-crit moves are 8x likelier
        double pCrit = clamp(def.getBaseStat(Stat.SPD) / 512.0 * (a.getCriticalHitRatio() > 1 ? 8 : 1), 0.0, 255.0 / 256.0);
        double critFactor = 1.0 + pCrit;
        double hits = EXPECTED_HITS.getOrDefault(name, 1.0);
        return 0.925 * base * critFactor * eff * screen * acc * hits / defHp;
    }

    public static double bestDamageFrac(PokemonView atk, PokemonView def, TeamView defTeam)
    {
        if (atk == null || def == null || atk.hasFainted()) {
            return 0.0;
        }
        double best = 0.0;
        for (MoveView m : atk.getAvailableMoves()) {
            best = Math.max(best, expectedDamageFrac(m, atk, def, defTeam));
        }
        return best;
    }

    public static double bestEffectiveness(PokemonView atk, PokemonView def)
    {
        if (atk == null || def == null) {
            return 1.0;
        }
        double best = 0.0;
        boolean any = false;
        for (MoveView m : atk.getAvailableMoves()) {
            if (m == null || m.getCategory() == Move.Category.STATUS || m.getPower() == null) {
                continue;
            }
            any = true;
            best = Math.max(best, effectiveness(m.getType(), def));
        }
        return any ? best : 1.0;
    }

    public static double effectiveness(Type attacking, PokemonView def)
    {
        if (attacking == null || def == null) {
            return 1.0;
        }
        double m = 1.0;
        Type t1 = def.getCurrentType1();
        Type t2 = def.getCurrentType2();
        if (t1 != null) {
            m *= TYPE_CHART[attacking.ordinal()][t1.ordinal()];
        }
        if (t2 != null && t2 != t1) {
            m *= TYPE_CHART[attacking.ordinal()][t2.ordinal()];
        }
        return m;
    }

    /** Maps {0, .25, .5, 1, 2, 4} onto [-1, 1]; immunity maps to -1. */
    private static double log2Eff(double eff)
    {
        if (eff <= 0.0) {
            return -1.0;
        }
        return clamp(Math.log(eff) / Math.log(2.0) / 2.0, -1.0, 1.0);
    }

    public static double effectiveSpeed(PokemonView p)
    {
        if (p == null) {
            return 0.0;
        }
        double s = p.getCurrentStat(Stat.SPD);
        if (p.getNonVolatileStatus() == NonVolatileStatus.PARALYSIS) {
            s *= 0.75;
        }
        return s;
    }

    private static double[][] buildTypeChart()
    {
        double[][] chart = new double[NT][NT];
        for (int i = 0; i < NT; i++) {
            for (int j = 0; j < NT; j++) {
                chart[i][j] = Type.getEffectivenessModifier(ALL_TYPES[i], ALL_TYPES[j]);
            }
        }
        return chart;
    }

    private static void writeTypes(Writer w, PokemonView p)
    {
        Type t1 = p.getCurrentType1();
        Type t2 = p.getCurrentType2();
        for (Type t : ALL_TYPES) {
            w.add(t == t1 || t == t2 ? 1.0 : 0.0);
        }
    }

    private static void writeStatusOneHot(Writer w, NonVolatileStatus s)
    {
        w.add(s == NonVolatileStatus.NONE ? 1.0 : 0.0);
        w.add(s == NonVolatileStatus.POISON ? 1.0 : 0.0);
        w.add(s == NonVolatileStatus.TOXIC ? 1.0 : 0.0);
        w.add(s == NonVolatileStatus.PARALYSIS ? 1.0 : 0.0);
        w.add(s == NonVolatileStatus.BURN ? 1.0 : 0.0);
        w.add(s == NonVolatileStatus.SLEEP ? 1.0 : 0.0);
        w.add(s == NonVolatileStatus.FREEZE ? 1.0 : 0.0);
    }

    private static double hpFrac(PokemonView p)
    {
        int max = p.getInitialStat(Stat.HP);
        return max <= 0 ? 0.0 : clamp(p.getCurrentStat(Stat.HP) / (double) max, 0.0, 1.0);
    }

    private static double teamHpFrac(TeamView team)
    {
        double cur = 0.0, max = 0.0;
        for (int i = 0; i < team.size(); i++) {
            PokemonView p = team.getPokemonView(i);
            if (p == null) continue;
            cur += Math.max(0, p.getCurrentStat(Stat.HP));
            max += p.getInitialStat(Stat.HP);
        }
        return max <= 0 ? 0.0 : cur / max;
    }

    private static int countAlive(TeamView team)
    {
        int n = 0;
        for (int i = 0; i < team.size(); i++) {
            PokemonView p = team.getPokemonView(i);
            if (p != null && !p.hasFainted()) n++;
        }
        return n;
    }

    private static double normCounter(int turns, int cap)
    {
        return clamp(turns / (double) cap, 0.0, 1.0);
    }

    private static double clamp(double v, double lo, double hi)
    {
        return Math.max(lo, Math.min(hi, v));
    }

    private static final class Writer
    {
        final double[] data;
        int pos = 0;

        Writer(int n) { this.data = new double[n]; }

        void add(double v)
        {
            if (pos < data.length) {
                data[pos] = v;
            }
            pos++; // keep counting past the end so a size mismatch is reported, not silently truncated
        }

        void skip(int n) { pos += n; }
    }
}
