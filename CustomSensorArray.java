package src.pas.pokemon.senses;


// SYSTEM IMPORTS
import java.util.List;
import java.util.ArrayList;

// JAVA PROJECT IMPORTS
import edu.bu.pas.pokemon.agents.senses.SensorArray;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.Pokemon.PokemonView;
import edu.bu.pas.pokemon.core.Team.TeamView;
import edu.bu.pas.pokemon.core.enums.NonVolatileStatus;
import edu.bu.pas.pokemon.core.enums.Stat;
import edu.bu.pas.pokemon.core.enums.Height;
import edu.bu.pas.pokemon.core.enums.Type;
import edu.bu.pas.pokemon.core.enums.Flag;
import edu.bu.pas.pokemon.linalg.Matrix;


public class CustomSensorArray
    extends SensorArray
{

    // TODO: make fields if you want!

    public CustomSensorArray()
    {
        // TODO: intialize those fields if you make any!
    }

    public Matrix getSensorValues(final BattleView state, final MoveView action)
    {
		final TeamView myTeam = state.getTeam1View();
		final TeamView oppTeam = state.getTeam2View();
		final PokemonView myActive = myTeam.getActivePokemonView();
		final PokemonView oppActive = oppTeam.getActivePokemonView();

		final List<Double> feats = new ArrayList<>();

		feats.add(safeRatio(myActive.getCurrentStat(Stat.HP), myActive.getInitialStat(Stat.HP)));
		feats.add(safeRatio(oppActive.getCurrentStat(Stat.HP), oppActive.getInitialStat(Stat.HP)));

		feats.add(safeRatio(myActive.getLastDamageDealt(), myActive.getInitialStat(Stat.HP)));
		feats.add(safeRatio(oppActive.getLastDamageDealt(), oppActive.getInitialStat(Stat.HP)));

		writeStatusOneHot(feats, myActive.getNonVolatileStatus());
		writeStatusOneHot(feats, oppActive.getNonVolatileStatus());

		feats.add(normCounter(myActive.getNonVolatileStatusCounter(NonVolatileStatus.SLEEP), 7));
		feats.add(normCounter(myActive.getNonVolatileStatusCounter(NonVolatileStatus.FREEZE), 7));
		feats.add(normCounter(oppActive.getNonVolatileStatusCounter(NonVolatileStatus.SLEEP), 7));
		feats.add(normCounter(oppActive.getNonVolatileStatusCounter(NonVolatileStatus.FREEZE), 7));

		feats.addAll(stageMultipliers(myActive));
		feats.addAll(stageMultipliers(oppActive));

		final int mySpd = myActive.getCurrentStat(Stat.SPD);
		final int oppSpd = oppActive.getCurrentStat(Stat.SPD);
		feats.add(mySpd > oppSpd ? 1.0 : 0.0);

		final int mySize = myTeam.size();
		final int oppSize = oppTeam.size();
		final int myFainted = countFainted(myTeam);
		final int oppFainted = countFainted(oppTeam);
		feats.add(safeRatio(mySize, 6));
		feats.add(safeRatio(myFainted, Math.max(1, mySize)));
		feats.add(safeRatio(mySize - myFainted, Math.max(1, mySize)));
		feats.add(safeRatio(oppSize, 6));
		feats.add(safeRatio(oppFainted, Math.max(1, oppSize)));
		feats.add(safeRatio(oppSize - oppFainted, Math.max(1, oppSize)));

		feats.add(normCounter(myTeam.getNumReflectTurnsRemaining(), 5));
		feats.add(normCounter(myTeam.getNumLightScreenTurnsRemaining(), 5));
		feats.add(normCounter(oppTeam.getNumReflectTurnsRemaining(), 5));
		feats.add(normCounter(oppTeam.getNumLightScreenTurnsRemaining(), 5));

		if (action != null) {
			final Type at = action.getType();
			final boolean stab = at != null &&
				(at == myActive.getCurrentType1() || at == myActive.getCurrentType2());
			feats.add(stab ? 1.0 : 0.0);
		} else {
			feats.add(0.0);
		}

		writeActionFeatures(feats, action, myActive, oppActive);
        writePokemonTypeOneHot(feats, myActive);
        writePokemonTypeOneHot(feats, oppActive);
        writeMoveTypeOneHot(feats, action);

        writeVolatileFlags(feats, myActive);
        writeVolatileFlags(feats, oppActive);
        writeHeightOneHot(feats, myActive);
        writeHeightOneHot(feats, oppActive);

        writeBenchInfo(feats, myTeam, true);
        writeBenchInfo(feats, oppTeam, true);
        writeTeamTypeDistribution(feats, myTeam);
        writeTeamTypeDistribution(feats, oppTeam);
        writeTeamStatusHistogram(feats, myTeam);
        writeTeamStatusHistogram(feats, oppTeam);

		final Matrix out = Matrix.full(1, feats.size(), 0.0);
		for (int i = 0; i < feats.size(); i++) {
			out.set(0, i, feats.get(i));
		}
		return out;
    }




    private int countFainted(TeamView team){
        int count = 0;
        for (int i =0; i < team.size(); i++){
            if (team.getPokemonView(i).hasFainted()){
                count++;
            }
        }
        return count;
    }

    private List<Double> stageMultipliers(PokemonView pokemon){
        List<Double> multipliers = new ArrayList<>();
        multipliers.add((double) pokemon.getStatMultiplier(Stat.ATK) / 6.0);
        multipliers.add((double) pokemon.getStatMultiplier(Stat.DEF) / 6.0);
        multipliers.add((double) pokemon.getStatMultiplier(Stat.SPD) / 6.0);
        multipliers.add((double) pokemon.getStatMultiplier(Stat.SPATK) / 6.0);
        multipliers.add((double) pokemon.getStatMultiplier(Stat.SPDEF) / 6.0);
        multipliers.add((double) pokemon.getStatMultiplier(Stat.ACC) / 6.0);
        multipliers.add((double) pokemon.getStatMultiplier(Stat.EVASIVE) / 6.0);
        return multipliers;
    }

	private static double safeRatio(final int num, final int den){
		return den <= 0 ? 0.0 : Math.max(0.0, Math.min(1.0, ((double) num) / den));
	}

	private static double normCounter(final int turns, final int cap){
		final int bounded = Math.max(0, Math.min(turns, cap));
		return safeRatio(bounded, cap);
	}


	private static void writeStatusOneHot(final List<Double> dst, final NonVolatileStatus s){
		dst.add(s == NonVolatileStatus.NONE ? 1.0 : 0.0);
		dst.add(s == NonVolatileStatus.POISON ? 1.0 : 0.0);
		dst.add(s == NonVolatileStatus.TOXIC ? 1.0 : 0.0);
		dst.add(s == NonVolatileStatus.PARALYSIS ? 1.0 : 0.0);
		dst.add(s == NonVolatileStatus.BURN ? 1.0 : 0.0);
		dst.add(s == NonVolatileStatus.SLEEP ? 1.0 : 0.0);
		dst.add(s == NonVolatileStatus.FREEZE ? 1.0 : 0.0);
	}

    private static void writePokemonTypeOneHot(List<Double> dst, PokemonView p) {
        Type t1 = p.getCurrentType1();
        Type t2 = p.getCurrentType2();
        Type[] all = Type.values();
        for (Type t : all) {
            boolean isThisType = (t == t1) || (t == t2);
            dst.add(isThisType ? 1.0 : 0.0);
        }
    }

    private static void writeMoveTypeOneHot(List<Double> dst, MoveView action) {
        Type[] all = Type.values();
        Type mt = (action == null) ? null : action.getType();
        for (Type t : all) {
            dst.add(mt != null && mt == t ? 1.0 : 0.0);
        }
    }

    private static void writeVolatileFlags(List<Double> dst, PokemonView p) {
        dst.add(p.getFlag(Flag.CONFUSED) ? 1.0 : 0.0);
        dst.add(p.getFlag(Flag.TRAPPED) ? 1.0 : 0.0);
        dst.add(p.getFlag(Flag.SEEDED) ? 1.0 : 0.0);
        dst.add(p.getFlag(Flag.FLINCHED) ? 1.0 : 0.0);
        dst.add(p.getFlag(Flag.FOCUS_ENERGY) ? 1.0 : 0.0);
        dst.add(p.getSubstitute() != null ? 1.0 : 0.0);
    }
    
    private static void writeHeightOneHot(List<Double> dst, PokemonView p) {
        Height h = p.getHeight();
        for (Height val : Height.values()) {
            dst.add(h == val ? 1.0 : 0.0);
        }
    }

    private void writeBenchInfo(List<Double> dst, TeamView team, boolean includeStatuses) {
        int size = team.size();
        int activeIdx = team.getActivePokemonIdx();
        for (int i = 0; i < 6; i++) {
            if (i >= size) {
                dst.add(0.0); // placeholder for isActive
                dst.add(0.0);
                dst.add(0.0);
                if (includeStatuses) {
                    for (int s = 0; s < 7; s++) {
                        dst.add(0.0);
                    }
                }
                continue;
            }
            PokemonView p = team.getPokemonView(i);
            dst.add(i == activeIdx ? 1.0 : 0.0);
            dst.add(p.hasFainted() ? 1.0 : 0.0);
            dst.add(safeRatio(p.getCurrentStat(Stat.HP), p.getInitialStat(Stat.HP)));
            if (includeStatuses) {
                writeStatusOneHot(dst, p.getNonVolatileStatus());
            }
        }
    }

    private void writeTeamTypeDistribution(List<Double> dst, TeamView team) {
        Type[] allTypes = Type.values();
        double[] counts = new double[allTypes.length];
        for (int i = 0; i < team.size(); i++) {
            PokemonView p = team.getPokemonView(i);
            if (p == null) {
                continue;
            }
            Type t1 = p.getCurrentType1();
            Type t2 = p.getCurrentType2();
            for (int tIdx = 0; tIdx < allTypes.length; tIdx++) {
                Type candidate = allTypes[tIdx];
                if (candidate == t1 || candidate == t2) {
                    counts[tIdx] += 1.0;
                }
            }
        }

        double normalizer = Math.max(1.0, team.size());
        for (double count : counts) {
            dst.add(count / normalizer);
        }
    }

    private void writeTeamStatusHistogram(List<Double> dst, TeamView team) {
        double[] counts = new double[7];
        for (int i = 0; i < team.size(); i++) {
            PokemonView p = team.getPokemonView(i);
            NonVolatileStatus status = (p == null) ? NonVolatileStatus.NONE : p.getNonVolatileStatus();
            counts[statusIndex(status)] += 1.0;
        }
        double normalizer = Math.max(1.0, team.size());
        for (double count : counts) {
            dst.add(count / normalizer);
        }
    }

    private int statusIndex(NonVolatileStatus status) {
        switch (status) {
            case NONE:
                return 0;
            case POISON:
                return 1;
            case TOXIC:
                return 2;
            case PARALYSIS:
                return 3;
            case BURN:
                return 4;
            case SLEEP:
                return 5;
            case FREEZE:
                return 6;
            default:
                return 0;
        }
    }
    
    

    

	private static void writeActionFeatures(final List<Double> dst, final MoveView action, final PokemonView caster, final PokemonView defender){
		if (action == null){
			// category(3), type(0 placeholder), power, acc, ppRatio, priority, critRatio, canHitHeight
			dst.add(0.0); dst.add(0.0); dst.add(0.0);
			dst.add(0.0);
			dst.add(0.0);
			dst.add(0.0);
			dst.add(0.5);
			dst.add(0.5);
			dst.add(0.0);
			return;
		}

		final boolean isPhys = action.getCategory() != null && action.getCategory().name().equals("PHYSICAL");
		final boolean isSpec = action.getCategory() != null && action.getCategory().name().equals("SPECIAL");
		final boolean isStatus = action.getCategory() != null && action.getCategory().name().equals("STATUS");
		dst.add(isPhys ? 1.0 : 0.0);
		dst.add(isSpec ? 1.0 : 0.0);
		dst.add(isStatus ? 1.0 : 0.0);

		final Integer pow = action.getPower();
		final Integer acc = action.getAccuracy();
		dst.add(pow == null ? 0.0 : Math.max(0.0, Math.min(1.0, pow / 150.0)));
		dst.add(acc == null ? 1.0 : Math.max(0.0, Math.min(1.0, acc / 100.0)));

		final Integer pp = action.getPP();
		dst.add(pp == null ? 1.0 : Math.max(0.0, Math.min(1.0, pp / 40.0)));
		final int pr = action.getPriority();
		final double prNorm = pr <= -1 ? 0.0 : (pr >= 1 ? 1.0 : 0.5);
		dst.add(prNorm);
		final int cr = action.getCriticalHitRatio();
		dst.add(Math.max(0.0, Math.min(1.0, cr / 2.0)));

		final Height defH = defender.getHeight();
		final boolean canHit = action.getCanHitHeights() != null && action.getCanHitHeights().contains(defH);
		dst.add(canHit ? 1.0 : 0.0);
	}
    
}

