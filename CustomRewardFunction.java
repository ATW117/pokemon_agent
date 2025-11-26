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
        return -5.0;
    }

    public double getUpperBound()
    {
        return 5.0;
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
            statusReward += 0.3;  
        }
    if (oppStatusBefore == NonVolatileStatus.NONE && oppStatusAfter != NonVolatileStatus.NONE) {
        if (oppStatusAfter == NonVolatileStatus.SLEEP || oppStatusAfter == NonVolatileStatus.FREEZE) {
            statusReward += 0.6;
        } else if (oppStatusAfter == NonVolatileStatus.TOXIC) {
            statusReward += 0.5;  
        } else {
            statusReward += 0.25; 
        }
    }
    
    if (myStatusBefore == NonVolatileStatus.NONE && myStatusAfter != NonVolatileStatus.NONE) {
        if (myStatusAfter == NonVolatileStatus.SLEEP || myStatusAfter == NonVolatileStatus.FREEZE) {
            statusReward -= 0.6;
        } else if (myStatusAfter == NonVolatileStatus.TOXIC) {
            statusReward -= 0.5;
        } else {
            statusReward -= 0.25;
        }
    }
        
    
        double myHpBefore  = hpFrac(myTeamBefore);
        double oppHpBefore = hpFrac(oppTeamBefore);
        double myHpAfter   = hpFrac(myTeamAfter);
        double oppHpAfter  = hpFrac(oppTeamAfter);
    
        double hpReward = 1.5 * ((myHpAfter - oppHpAfter) - (myHpBefore - oppHpBefore));
    
        int myFaintBefore  = faintedCount(myTeamBefore);
        int oppFaintBefore = faintedCount(oppTeamBefore);
        int myFaintAfter   = faintedCount(myTeamAfter);
        int oppFaintAfter  = faintedCount(oppTeamAfter);
    
        int myFaintDelta  = myFaintAfter  - myFaintBefore;
        int oppFaintDelta = oppFaintAfter - oppFaintBefore;
    
        double koReward = 0.0;
        if (oppFaintDelta > 0) koReward += 2.0 * oppFaintDelta;
        if (myFaintDelta  > 0) koReward -= 2.0 * myFaintDelta;
    
        double terminalReward = 0.0;
        if (nextState.isOver()) {
            boolean iLost  = (myFaintAfter  == myTeamAfter.size());
            boolean oppLost = (oppFaintAfter == oppTeamAfter.size());
        
            if (iLost && !oppLost) {
                terminalReward -= 4.0;
            } else if (!iLost && oppLost) {
                terminalReward += 4.0;
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