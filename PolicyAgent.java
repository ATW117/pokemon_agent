package src.pas.pokemon.agents;


// SYSTEM IMPORTS
import net.sourceforge.argparse4j.inf.Namespace;

import edu.bu.pas.pokemon.agents.NeuralQAgent;
import edu.bu.pas.pokemon.agents.senses.SensorArray;
import edu.bu.pas.pokemon.agents.rewards.RewardFunction;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.linalg.Matrix;
import edu.bu.pas.pokemon.nn.Model;
import edu.bu.pas.pokemon.nn.models.Sequential;
import edu.bu.pas.pokemon.nn.layers.Dense; // fully connected layer
import edu.bu.pas.pokemon.nn.layers.ReLU;  // some activations (below too)
import edu.bu.pas.pokemon.nn.layers.Tanh;
import edu.bu.pas.pokemon.nn.layers.Sigmoid;
import edu.bu.pas.pokemon.core.Move;
import edu.bu.pas.pokemon.training.data.ReplayBuffer;
import edu.bu.pas.pokemon.core.Team.TeamView;
import edu.bu.pas.pokemon.core.Pokemon.PokemonView;
import edu.bu.pas.pokemon.core.enums.Stat;
import edu.bu.pas.pokemon.core.enums.NonVolatileStatus;
import edu.bu.pas.pokemon.core.enums.Type;
import java.util.Random;
import java.util.List;
import java.util.ArrayList;

// JAVA PROJECT IMPORTS
import src.pas.pokemon.senses.CustomSensorArray;


public class PolicyAgent
    extends NeuralQAgent
{
    private static final int EPSILON_LOG_INTERVAL = 100000;

    private Random random;
    private double epsilonStart;
    private double epsilonEnd;
    private int epsilonDecaySteps;
    private boolean isTraining; 
    
    private Model targetNetwork;
    private int targetUpdateFrequency; 
    private int targetUpdateCounter; 
    private int gamesPlayed = 0;
    private int gamesWon = 0;
    private static int globalTrainingStepCount = 0;
    
    public PolicyAgent()
    {
        super();
        this.random = new Random();
        this.epsilonStart = 1.0;
        this.epsilonEnd = 0.1;
        this.epsilonDecaySteps = 1000000;
        this.isTraining = true; 
        this.targetNetwork = null;
        this.targetUpdateFrequency = 1000; 
        this.targetUpdateCounter = 0;
        
    }
    
    private double getCurrentEpsilon() {
        double progress = Math.min(1.0, (double)globalTrainingStepCount / epsilonDecaySteps);
    double epsilon = epsilonEnd + (epsilonStart - epsilonEnd) * (1.0 - progress);
    return epsilon;
    }

    public void initializeSenses(Namespace args)
    {
        SensorArray modelSenses = new CustomSensorArray();
        this.setSensorArray(modelSenses);
    }

    @Override
    public void initialize(Namespace args)
    {
        // make sure you call this, this will call your initModel() and set a field
        // AND if the command line argument "inFile" is present will attempt to set
        // your model with the contents of that file.
        super.initialize(args);

        if (this.getModel() == null) {
            Model model = this.initModel();
            this.setModel(model);
        }
        this.initializeSenses(args);

        if (this.targetNetwork == null) {
            this.targetNetwork = this.initModel();
        }
        this.updateTargetNetwork(); 
        
    }

    private void updateTargetNetwork() {
        if (targetNetwork == null || this.getModel() == null) {
            return;
        }
        
        try {
            String tempFile = "/tmp/target_network_copy_" + System.currentTimeMillis() + ".model";
            this.getModel().save(tempFile);
            targetNetwork.load(tempFile);
            try {
                java.io.File temp = new java.io.File(tempFile);
                if (temp.exists()) {
                    temp.delete();
                }
            } catch (Exception e) {
            }
        } catch (Exception e) {
            System.err.println("Error updating target network: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Override
    public Model initModel()
    {
        final int N_INPUT = 284;
        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(N_INPUT, 128));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(128, 128));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(128, 1));
        return qFunction;
    }

    @Override
    public Integer chooseNextPokemon(BattleView view)
    {
        TeamView myTeam = this.getMyTeamView(view);
        TeamView oppTeam = this.getOpponentTeamView(view);
        PokemonView oppActive = (oppTeam == null) ? null : oppTeam.getActivePokemonView();
        Integer bestIdx = null;
        Double bestScore = Double.NEGATIVE_INFINITY;


        for (int i = 0; i < myTeam.size(); i++){
            PokemonView myPokemon = myTeam.getPokemonView(i);
            if (myPokemon.hasFainted()){
                continue;
            }

            double score = 0.0;
            double hpFrac = (double) myPokemon.getCurrentStat(Stat.HP) / Math.max(1.0, (double) myPokemon.getInitialStat(Stat.HP));

            score += 1.8 * hpFrac;
            score += 0.4 * getNormalizedStat(myPokemon, Stat.SPD);

            if (oppActive != null) {
                double offensiveOutlook = estimateOffensivePotential(myPokemon, oppActive);
                double defensiveRisk = estimateDefensiveRisk(myPokemon, oppActive);
                score += 0.9 * offensiveOutlook;
                score -= 0.7 * defensiveRisk;

                if (myPokemon.getCurrentStat(Stat.SPD) > oppActive.getCurrentStat(Stat.SPD)) {
                    score += 0.2;
                }
            }

            score -= statusPenalty(myPokemon.getNonVolatileStatus());

            if (score > bestScore){
                bestScore = score;
                bestIdx = i;
            }

    }

    return bestIdx;
    }

    private double statusPenalty(NonVolatileStatus status) {
        if (status == null || status == NonVolatileStatus.NONE) {
            return 0.0;
        }
        switch (status) {
            case SLEEP:
            case FREEZE:
                return 0.9;
            case PARALYSIS:
                return 0.6;
            case BURN:
                return 0.5;
            case POISON:
            case TOXIC:
                return 0.4;
            default:
                return 0.3;
        }
    }

    private double estimateOffensivePotential(PokemonView attacker, PokemonView defender) {
        if (attacker == null || defender == null) {
            return 0.0;
        }

        List<MoveView> moves = attacker.getAvailableMoves();
        if (moves == null || moves.isEmpty()) {
            return 0.0;
        }

        double bestScore = 0.0;
        for (MoveView move : moves) {
            if (move == null || move.getType() == null) {
                continue;
            }
            double typeMultiplier = computeTypeMultiplier(move.getType(), defender);
            double stabBonus = isStab(move.getType(), attacker) ? 0.2 : 0.0;
            double powerFactor = normalizePower(move.getPower());
            double moveScore = typeMultiplier * (0.5 + powerFactor) + stabBonus;
            bestScore = Math.max(bestScore, moveScore);
        }
        return clamp(bestScore, 0.0, 2.5);
    }

    private double estimateDefensiveRisk(PokemonView candidate, PokemonView opponent) {
        if (candidate == null || opponent == null) {
            return 0.0;
        }

        List<MoveView> moves = opponent.getAvailableMoves();
        if (moves == null || moves.isEmpty()) {
            return 0.0;
        }

        double worstThreat = 0.0;
        for (MoveView move : moves) {
            if (move == null || move.getType() == null) {
                continue;
            }
            double typeMultiplier = computeTypeMultiplier(move.getType(), candidate);
            double powFactor = 0.5 + normalizePower(move.getPower());
            worstThreat = Math.max(worstThreat, typeMultiplier * powFactor);
        }
        return clamp(worstThreat, 0.0, 3.0);
    }

    private boolean isStab(Type moveType, PokemonView pokemon) {
        if (moveType == null || pokemon == null) {
            return false;
        }
        return moveType == pokemon.getCurrentType1() || moveType == pokemon.getCurrentType2();
    }

    private double computeTypeMultiplier(Type attackingType, PokemonView defender) {
        if (attackingType == null || defender == null) {
            return 1.0;
        }
        double multiplier = 1.0;
        Type defType1 = defender.getCurrentType1();
        Type defType2 = defender.getCurrentType2();

        if (defType1 != null) {
            multiplier *= Type.getEffectivenessModifier(attackingType, defType1);
        }
        if (defType2 != null && defType2 != defType1) {
            multiplier *= Type.getEffectivenessModifier(attackingType, defType2);
        }
        return multiplier;
    }

    private double normalizePower(Integer power) {
        if (power == null) {
            return 0.3;
        }
        return clamp(power / 120.0, 0.0, 1.0);
    }

    private double getNormalizedStat(PokemonView pokemon, Stat stat) {
        if (pokemon == null || stat == null) {
            return 0.0;
        }
        double base = Math.max(1.0, (double) pokemon.getInitialStat(stat));
        return clamp(pokemon.getCurrentStat(stat) / base, 0.0, 1.5);
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }
    @Override
    public MoveView getMove(BattleView view)
    {
        if (this.getSensorArray() == null) {
            this.initializeSenses(null);
        }
        if (this.getModel() == null) {
            try {
                Model model = this.initModel();
                this.setModel(model);
                if (this.targetNetwork == null) {
                    this.targetNetwork = this.initModel();
                    this.updateTargetNetwork();
                }
            } catch (Exception e) {
                System.err.println("Error initializing model in getMove: " + e.getMessage());
                TeamView myTeam = this.getMyTeamView(view);
                if (myTeam != null) {
                    PokemonView active = myTeam.getActivePokemonView();
                    if (active != null) {
                        List<MoveView> moves = active.getAvailableMoves();
                        if (moves != null && !moves.isEmpty()) {
                            return moves.get(0);
                        }
                    }
                }
                return null;
            }
        }
        
        if (isTraining) {
            globalTrainingStepCount++;
            logEpsilonIfNeeded();
        }
        
        TeamView myTeam = this.getMyTeamView(view);
        if (myTeam == null) {
            return null;
        }
        
        PokemonView active = myTeam.getActivePokemonView();
        if (active == null) {
            return null;
        }
        
        List<MoveView> legalMoves = active.getAvailableMoves();
        if (legalMoves == null || legalMoves.isEmpty()) {
            return this.argmax(view);
        }
        
        if (isTraining) {
            double epsilon = getCurrentEpsilon();
            if (random.nextDouble() < epsilon) {
                return legalMoves.get(random.nextInt(legalMoves.size()));
            }
        }
        
        return this.argmax(view);
    }
    
    public void setTrainingMode(boolean training) {
        this.isTraining = training;
    }
    @Override
    public Matrix makeGroundTruth(ReplayBuffer buffer, double discountFactor, RewardFunction rewardFunction)
    {
        targetUpdateCounter++;
        if (targetUpdateCounter >= targetUpdateFrequency) {
            updateTargetNetwork();
            targetUpdateCounter = 0;
        }
        
        Matrix ygroundTruth = Matrix.zeros(buffer.size(), 1);
        for (int rIdx = 0; rIdx < buffer.size(); rIdx++){
            BattleView state = buffer.getState(rIdx);
            MoveView action = buffer.getAction(rIdx);
            BattleView nextState = buffer.getNextState(rIdx);
            
            double reward = rewardFunction.getStateActionStateReward(state, action, nextState);
            
            if (nextState == null || nextState.isOver()){
                ygroundTruth.set(rIdx, 0, reward);
            }else{
                TeamView myTeam = this.getMyTeamView(nextState);
                PokemonView active = myTeam.getActivePokemonView();
                int numMoves = active.getAvailableMoves().size();
                List<MoveView> legalMoves = new ArrayList<>();
                for (int i = 0; i < numMoves; i++){
                    MoveView move = active.getAvailableMoves().get(i);
                    if (move != null){
                        legalMoves.add(move);
                    }
                }
                
                if (legalMoves.isEmpty()) {
                    ygroundTruth.set(rIdx, 0, reward);
                    continue;
                }
                
                Double maxQMain = null;
                MoveView bestAction = null;
                for (MoveView move : legalMoves) {
                    double qValMain = this.eval(nextState, move);
                    if (maxQMain == null || qValMain > maxQMain) {
                        maxQMain = qValMain;
                        bestAction = move;
                    }
                }
                
                double targetQValue = 0.0;
                if (bestAction != null && targetNetwork != null) {
                    try {
                        Matrix input = this.getModelInput(nextState, bestAction);
                        Matrix output = targetNetwork.forward(input);
                        targetQValue = output.get(0, 0);
                    } catch (Exception e) {
                        targetQValue = maxQMain;
                    }
                } else if (bestAction != null) {
                    targetQValue = maxQMain;
                }
                
                ygroundTruth.set(rIdx, 0, reward + discountFactor * targetQValue);
            }
        }
        return ygroundTruth;
    }

    @Override
    public Move.MoveView argmax(BattleView view)
    {
        if (this.getSensorArray() == null) {
            this.initializeSenses(null);
        }
        if (this.getModel() == null) {
            try {
                Model model = this.initModel();
                this.setModel(model);
                if (this.targetNetwork == null) {
                    this.targetNetwork = this.initModel();
                    this.updateTargetNetwork();
                }
            } catch (Exception e) {
                System.err.println("Error initializing model in argmax: " + e.getMessage());
                TeamView myTeam = this.getMyTeamView(view);
                if (myTeam != null) {
                    PokemonView active = myTeam.getActivePokemonView();
                    if (active != null) {
                        List<MoveView> moves = active.getAvailableMoves();
                        if (moves != null && !moves.isEmpty()) {
                            return moves.get(0);
                        }
                    }
                }
                return null;
            }
        }
        
        TeamView myTeam = this.getMyTeamView(view);
        PokemonView active = myTeam.getActivePokemonView();
        if (active == null) {
            return null;
        }
        
        List<MoveView> availableMoves = active.getAvailableMoves();
        if (availableMoves == null || availableMoves.isEmpty()) {
            return null;
        }
        
        List<MoveView> legalMoves = new ArrayList<>();
        for (MoveView move : availableMoves) {
            if (move != null) {
                legalMoves.add(move);
            }
        }
        
        if (legalMoves.isEmpty()) {
            return null;
        }
        
        Double maxVal = null;
        MoveView bestMove = null;
        
        for (MoveView move : legalMoves) {
            try {
                double qVal = this.eval(view, move);
                if (maxVal == null || qVal > maxVal) {
                    maxVal = qVal;
                    bestMove = move;
                }
            } catch (Exception e) {
                System.err.println("Warning: eval failed for move: " + e.getMessage());
                continue;
            }
        }
        
        if (bestMove == null && !legalMoves.isEmpty()) {
            return legalMoves.get(0);
        }
        
        return bestMove;
    }

    @Override
    public void train(){
        setTrainingMode(true);
    }
    
    @Override
    public void eval(){
        setTrainingMode(false);
    }


    @Override
    public void afterGameEnds(BattleView view)
    {
        gamesPlayed++;
    
    TeamView myTeam = this.getMyTeamView(view);
    TeamView oppTeam = this.getOpponentTeamView(view);
    
    int myFainted = 0;
    int oppFainted = 0;
    
    for (int i = 0; i < myTeam.size(); i++) {
        if (myTeam.getPokemonView(i).hasFainted()) myFainted++;
    }
    for (int i = 0; i < oppTeam.size(); i++) {
        if (oppTeam.getPokemonView(i).hasFainted()) oppFainted++;
    }
    
    if (oppFainted == oppTeam.size() && myFainted < myTeam.size()) {
        gamesWon++;
    }
    
        // Log every 100 games
        if (gamesPlayed % 100 == 0) {
        double winRate = (double) gamesWon / gamesPlayed;
        System.err.println(String.format("Games: %d | Win Rate: %.3f | Epsilon: %.3f", 
            gamesPlayed, winRate, getCurrentEpsilon()));
    }
    }

    private void logEpsilonIfNeeded() {
        if (!isTraining) {
            return;
        }
        if (globalTrainingStepCount % EPSILON_LOG_INTERVAL != 0) {
            return;
        }
        double epsilon = getCurrentEpsilon();
        System.err.println(String.format("[EPSILON] steps=%d epsilon=%.6f decaySteps=%d start=%.2f end=%.2f", 
            globalTrainingStepCount, epsilon, epsilonDecaySteps, epsilonStart, epsilonEnd));
    }
}