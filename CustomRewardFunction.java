package src.pas.pokemon.rewards;


// SYSTEM IMPORTS


// JAVA PROJECT IMPORTS
import edu.bu.pas.pokemon.agents.rewards.RewardFunction;
import edu.bu.pas.pokemon.agents.rewards.RewardFunction.RewardType;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.Pokemon.PokemonView;
import edu.bu.pas.pokemon.core.SwitchMove.SwitchMoveView;
import edu.bu.pas.pokemon.core.Team.TeamView;
import edu.bu.pas.pokemon.core.enums.Stat;
import edu.bu.pas.pokemon.core.enums.NonVolatileStatus;

/**
 * Shaped reward on roughly a [-1, 1] scale per step so Q-targets stay small:
 *   - change in (my team HP fraction - opponent team HP fraction)
 *   - +/- KO_REWARD per faint
 *   - per-pokemon status changes (compared slot by slot, so switching never looks like a cure)
 *   - +/- 1 for winning/losing, -0.5 for a mutual KO
 *   - a small penalty only for *voluntary* switches
 * The perspective is the team index set by the framework/trainer via setTeamIdx (defaults to team 0).
 */
public class CustomRewardFunction
    extends RewardFunction
{
    public static final double HP_WEIGHT = 1.0;
    public static final double KO_REWARD = 0.15;
    public static final double WIN_REWARD = 1.0;
    public static final double CURE_REWARD = 0.03;
    public static final double SWITCH_PENALTY = 0.02;
    public static final double DRAW_REWARD = -0.5;

    public CustomRewardFunction()
    {
        super(RewardType.STATE_ACTION_STATE);
    }

    public double getLowerBound()
    {
        return -2.0;
    }

    public double getUpperBound()
    {
        return 2.0;
    }

    public double getStateReward(final BattleView state)
    {
        return 0d;
    }

    public double getStateActionReward(final BattleView state,
                                       final MoveView action)
    {
        return 0d;
    }

    public double getStateActionStateReward(final BattleView state,
                                            final MoveView action,
                                            final BattleView nextState)
    {
        return reward(state, action, nextState, this.getTeamIdx() == null ? 0 : this.getTeamIdx());
    }

    public static double reward(final BattleView state, final MoveView action, final BattleView nextState, final int me)
    {
        if (state == null || nextState == null) {
            return 0.0;
        }
        TeamView myBefore = state.getTeamView(me);
        TeamView oppBefore = state.getTeamView(1 - me);
        TeamView myAfter = nextState.getTeamView(me);
        TeamView oppAfter = nextState.getTeamView(1 - me);

        double hp = HP_WEIGHT * ((hpFrac(myAfter) - hpFrac(oppAfter)) - (hpFrac(myBefore) - hpFrac(oppBefore)));

        int myKo = fainted(myAfter) - fainted(myBefore);
        int oppKo = fainted(oppAfter) - fainted(oppBefore);
        double ko = KO_REWARD * (oppKo - myKo);

        double status = statusDelta(oppBefore, oppAfter) - statusDelta(myBefore, myAfter);

        double terminal = 0.0;
        if (nextState.isOver()) {
            boolean iLost = alive(myAfter) == 0;
            boolean oppLost = alive(oppAfter) == 0;
            if (oppLost && !iLost) terminal = WIN_REWARD;
            else if (iLost && !oppLost) terminal = -WIN_REWARD;
            else if (iLost && oppLost) terminal = DRAW_REWARD; // mutual KO is not a win
        }

        double switchPenalty = (action instanceof SwitchMoveView) ? -SWITCH_PENALTY : 0.0;

        double r = hp + ko + status + terminal + switchPenalty;
        return Math.max(-2.0, Math.min(2.0, r));
    }

    /** How much worse off a team's statuses got (positive = more/worse ailments), compared slot by slot. */
    private static double statusDelta(TeamView before, TeamView after)
    {
        double d = 0.0;
        int n = Math.min(before.size(), after.size());
        for (int i = 0; i < n; i++) {
            PokemonView b = before.getPokemonView(i);
            PokemonView a = after.getPokemonView(i);
            if (b == null || a == null || a.hasFainted()) {
                continue;
            }
            NonVolatileStatus sb = b.getNonVolatileStatus();
            NonVolatileStatus sa = a.getNonVolatileStatus();
            if (sb == sa) {
                continue;
            }
            if (sb == NonVolatileStatus.NONE) {
                d += severity(sa);
            } else if (sa == NonVolatileStatus.NONE) {
                d -= CURE_REWARD;
            } else {
                d += severity(sa) - severity(sb);
            }
        }
        return d;
    }

    private static double severity(NonVolatileStatus s)
    {
        if (s == null) return 0.0;
        switch (s) {
            case SLEEP:
            case FREEZE:
                return 0.10;
            case TOXIC:
                return 0.08;
            case PARALYSIS:
                return 0.06;
            case BURN:
            case POISON:
                return 0.04;
            default:
                return 0.0;
        }
    }

    private static double hpFrac(TeamView team)
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

    private static int fainted(TeamView team)
    {
        int n = 0;
        for (int i = 0; i < team.size(); i++) {
            PokemonView p = team.getPokemonView(i);
            if (p != null && p.hasFainted()) n++;
        }
        return n;
    }

    private static int alive(TeamView team)
    {
        int n = 0;
        for (int i = 0; i < team.size(); i++) {
            PokemonView p = team.getPokemonView(i);
            if (p != null && !p.hasFainted()) n++;
        }
        return n;
    }
}
