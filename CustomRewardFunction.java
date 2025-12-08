package src.pas.pokemon.rewards;


// SYSTEM IMPORTS


// JAVA PROJECT IMPORTS
import edu.bu.pas.pokemon.agents.rewards.RewardFunction;
import edu.bu.pas.pokemon.agents.rewards.RewardFunction.RewardType;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.Pokemon.PokemonView;
import edu.bu.pas.pokemon.core.Team.TeamView;
import edu.bu.pas.pokemon.core.enums.Stat;
import edu.bu.pas.pokemon.core.enums.NonVolatileStatus;

public class CustomRewardFunction
    extends RewardFunction
{

    public CustomRewardFunction()
    {
        super(RewardType.STATE_ACTION_STATE);
    }

    public double getLowerBound()
    {
        return -50.0;  // Scaled up 10x for better gradient signals and visualization
    }

    public double getUpperBound()
    {
        return 50.0;  // Scaled up 10x for better gradient signals and visualization
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
        TeamView myTeamBefore  = state.getTeam1View();
        TeamView oppTeamBefore = state.getTeam2View();
    
        TeamView myTeamAfter   = nextState.getTeam1View();
        TeamView oppTeamAfter  = nextState.getTeam2View();

        PokemonView myActiveBefore = myTeamBefore.getActivePokemonView();
        PokemonView oppActiveBefore = oppTeamBefore.getActivePokemonView();
        PokemonView myActiveAfter = myTeamAfter.getActivePokemonView();
        PokemonView oppActiveAfter = oppTeamAfter.getActivePokemonView();

        NonVolatileStatus myStatusBefore = myActiveBefore.getNonVolatileStatus();
        NonVolatileStatus oppStatusBefore = oppActiveBefore.getNonVolatileStatus();
        NonVolatileStatus myStatusAfter = myActiveAfter.getNonVolatileStatus();
        NonVolatileStatus oppStatusAfter = oppActiveAfter.getNonVolatileStatus();


        double statusReward = 0.0;
        if (myStatusBefore != NonVolatileStatus.NONE && myStatusAfter == NonVolatileStatus.NONE) {
            statusReward += 3.0;  // Scaled 10x: 0.3 -> 3.0
        }
    if (oppStatusBefore == NonVolatileStatus.NONE && oppStatusAfter != NonVolatileStatus.NONE) {
        if (oppStatusAfter == NonVolatileStatus.SLEEP || oppStatusAfter == NonVolatileStatus.FREEZE) {
            statusReward += 6.0;  // Scaled 10x: 0.6 -> 6.0
        } else if (oppStatusAfter == NonVolatileStatus.TOXIC) {
            statusReward += 5.0;  // Scaled 10x: 0.5 -> 5.0
        } else {
            statusReward += 2.5;  // Scaled 10x: 0.25 -> 2.5
        }
    }
    
    if (myStatusBefore == NonVolatileStatus.NONE && myStatusAfter != NonVolatileStatus.NONE) {
        if (myStatusAfter == NonVolatileStatus.SLEEP || myStatusAfter == NonVolatileStatus.FREEZE) {
            statusReward -= 6.0;  // Scaled 10x: -0.6 -> -6.0
        } else if (myStatusAfter == NonVolatileStatus.TOXIC) {
            statusReward -= 5.0;  // Scaled 10x: -0.5 -> -5.0
        } else {
            statusReward -= 2.5;  // Scaled 10x: -0.25 -> -2.5
        }
    }
        
    
        double myHpBefore  = hpFrac(myTeamBefore);
        double oppHpBefore = hpFrac(oppTeamBefore);
        double myHpAfter   = hpFrac(myTeamAfter);
        double oppHpAfter  = hpFrac(oppTeamAfter);
    
        double hpReward = 15.0 * ((myHpAfter - oppHpAfter) - (myHpBefore - oppHpBefore));  // Scaled 10x: 1.5 -> 15.0
    
        int myFaintBefore  = faintedCount(myTeamBefore);
        int oppFaintBefore = faintedCount(oppTeamBefore);
        int myFaintAfter   = faintedCount(myTeamAfter);
        int oppFaintAfter  = faintedCount(oppTeamAfter);
    
        int myFaintDelta  = myFaintAfter  - myFaintBefore;
        int oppFaintDelta = oppFaintAfter - oppFaintBefore;
    
        double koReward = 0.0;
        if (oppFaintDelta > 0) koReward += 20.0 * oppFaintDelta;  // Scaled 10x: 2.0 -> 20.0
        if (myFaintDelta  > 0) koReward -= 20.0 * myFaintDelta;   // Scaled 10x: -2.0 -> -20.0
    
        double terminalReward = 0.0;
        if (nextState.isOver()) {
            boolean iLost  = (myFaintAfter  == myTeamAfter.size());
            boolean oppLost = (oppFaintAfter == oppTeamAfter.size());
        
            if (iLost && !oppLost) {
                terminalReward -= 40.0;  // Scaled 10x: -4.0 -> -40.0
            } else if (!iLost && oppLost) {
                terminalReward += 40.0;   // Scaled 10x: +4.0 -> +40.0
            }else{
                terminalReward += 0.0;
            }
        }
    
        double reward = hpReward + koReward + terminalReward + statusReward;
    
        reward = Math.max(getLowerBound(), Math.min(getUpperBound(), reward));
    
        return reward;
    }

    private double hpFrac(TeamView team){
            double cur = 0.0;
            double max = 0.0;
            for (int i = 0; i < team.size(); i++) {
                PokemonView p = team.getPokemonView(i);
                int initHp = p.getInitialStat(Stat.HP);
                cur += p.getCurrentStat(Stat.HP);
                max += initHp;
            }
            if (max <= 0) return 0.0;
            return cur / max;
        }

        private int faintedCount(TeamView team) {
            int cnt = 0;
            for (int i = 0; i < team.size(); i++) {
                if (team.getPokemonView(i).hasFainted()) cnt++;
            }
            return cnt;
        }
        
}