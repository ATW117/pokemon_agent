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
import edu.bu.pas.pokemon.nn.Parameter;
import edu.bu.pas.pokemon.core.Move;
import edu.bu.pas.pokemon.core.SwitchMove;
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
    private int nextTargetUpdateStep;
    private int gamesPlayed = 0;
    private int gamesWon = 0;
    private static int globalTrainingStepCount = 0;
    
    private int consecutiveSwitchChoices = 0;
    
    public PolicyAgent()
    {
        super();
        this.random = new Random();
        this.epsilonStart = 1.0;
        this.epsilonEnd = 0.1;
        this.epsilonDecaySteps = 10000000;
        this.isTraining = false; 
        this.targetNetwork = null;
        this.targetUpdateFrequency = 10000; 
        this.nextTargetUpdateStep = this.targetUpdateFrequency;
        
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
            List<Parameter> mainParams = this.getModel().getParameters();
            List<Parameter> targetParams = targetNetwork.getParameters();
            
            if (mainParams.size() != targetParams.size()) {
                System.err.println("Warning: Parameter count mismatch in target network update");
                return;
            }
            
            for (int i = 0; i < mainParams.size(); i++) {
                Matrix mainValue = mainParams.get(i).getValue();
                Matrix targetValue = targetParams.get(i).getValue();
                
                int numRows = mainValue.getShape().getNumRows();
                int numCols = mainValue.getShape().getNumCols();
                targetValue.copySlice(0, numRows, 0, numCols, mainValue);
            }
        } catch (Exception e) {
            System.err.println("Error updating target network: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Override
    public Model initModel()
    {
        final int N_INPUT = 291;
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
    
    private List<MoveView> getAllLegalMoves(BattleView view) {
        TeamView myTeam = this.getMyTeamView(view);
        if (myTeam == null) {
            return new ArrayList<>();
        }
        
        PokemonView active = myTeam.getActivePokemonView();
        if (active == null) {
            return new ArrayList<>();
        }
        
        List<MoveView> legalMoves = new ArrayList<>(active.getAvailableMoves());
        
        int activeIdx = myTeam.getActivePokemonIdx();
        for (int i = 0; i < myTeam.size(); i++) {
            if (i != activeIdx) {
                PokemonView pokemon = myTeam.getPokemonView(i);
                if (pokemon != null && !pokemon.hasFainted()) {
                    SwitchMove switchMove = new SwitchMove(i);
                    legalMoves.add(switchMove.getView());
                }
            }
        }
        
        return legalMoves;
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
        PokemonView active = (myTeam != null) ? myTeam.getActivePokemonView() : null;
        List<MoveView> attackMoves = (active != null) ? active.getAvailableMoves() : new ArrayList<>();
        
        // During eval: only consider attacks (faster + no stalling)
        if (!isTraining && attackMoves != null && !attackMoves.isEmpty()) {
            return argmaxFromList(view, attackMoves);
        }
        
        List<MoveView> legalMoves = getAllLegalMoves(view);
        
        if (consecutiveSwitchChoices >= 2 && attackMoves != null && !attackMoves.isEmpty()) {
            List<MoveView> filteredMoves = new ArrayList<>();
            for (MoveView move : legalMoves) {
                if (attackMoves.contains(move)) {
                    filteredMoves.add(move);
                }
            }
            if (!filteredMoves.isEmpty()) {
                legalMoves = filteredMoves;
            }
        }
        
        if (legalMoves == null || legalMoves.isEmpty()) {
            return this.argmax(view);
        }
        
        MoveView chosen;
        if (isTraining) {
            double epsilon = getCurrentEpsilon();
            if (random.nextDouble() < epsilon) {
                chosen = legalMoves.get(random.nextInt(legalMoves.size()));
            } else {
                chosen = argmaxFromList(view, legalMoves);
            }
        } else {
            chosen = argmaxFromList(view, legalMoves);
        }
        
        boolean isSwitch = (chosen != null && attackMoves != null && !attackMoves.contains(chosen));
        if (isSwitch) {
            consecutiveSwitchChoices++;
        } else {
            consecutiveSwitchChoices = 0;
        }
        
        return chosen;
    }
    
    public void setTrainingMode(boolean training) {
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

    @Override
    public Matrix makeGroundTruth(ReplayBuffer buffer, double discountFactor, RewardFunction rewardFunction)
    {
        if (globalTrainingStepCount >= nextTargetUpdateStep) {
            System.err.println(String.format(
                "[TARGET_UPDATE] Steps=%d. Updating target network.", 
                globalTrainingStepCount
            ));
            
            updateTargetNetwork();
            nextTargetUpdateStep = globalTrainingStepCount + targetUpdateFrequency;
        }
        
        Matrix ygroundTruth = Matrix.zeros(buffer.size(), 1);
        Model networkToUse = (targetNetwork != null) ? targetNetwork : this.getModel();
        
        final int GROUND_TRUTH_BATCH_SIZE = 256; 

        List<Integer> stateIndices = new ArrayList<>(GROUND_TRUTH_BATCH_SIZE);
        List<Matrix> allInputs = new ArrayList<>();
        List<Integer> moveCountsPerState = new ArrayList<>(GROUND_TRUTH_BATCH_SIZE);

        for (int rIdx = 0; rIdx < buffer.size(); rIdx++){
            BattleView state = buffer.getState(rIdx);
            MoveView action = buffer.getAction(rIdx);
            BattleView nextState = buffer.getNextState(rIdx);
            
            double reward = rewardFunction.getStateActionStateReward(state, action, nextState);
            
            if (nextState == null || nextState.isOver()){
                ygroundTruth.set(rIdx, 0, reward);
                continue;
            }

            List<MoveView> legalMoves = getAllLegalMoves(nextState);
                
            if (legalMoves.isEmpty()) {
                ygroundTruth.set(rIdx, 0, reward);
                continue;
            }
                
            List<Matrix> currentMoveInputs = new ArrayList<>(legalMoves.size());
            for (MoveView move : legalMoves) {
                try {
                    Matrix input = this.getModelInput(nextState, move);
                    currentMoveInputs.add(input);
                } catch (Exception e) {
                }
            }

            if (!currentMoveInputs.isEmpty()) {
                stateIndices.add(rIdx);
                allInputs.addAll(currentMoveInputs);
                moveCountsPerState.add(currentMoveInputs.size());
            } else {
                ygroundTruth.set(rIdx, 0, reward);
            }
            
            boolean isLastSample = (rIdx == buffer.size() - 1);
            if ((stateIndices.size() >= GROUND_TRUTH_BATCH_SIZE || (isLastSample && !stateIndices.isEmpty()))) {
                processGroundTruthBatch(
                    ygroundTruth, buffer, rewardFunction, discountFactor, networkToUse,
                    stateIndices, allInputs, moveCountsPerState
                );
                stateIndices.clear();
                allInputs.clear();
                moveCountsPerState.clear();
            }
        }
        return ygroundTruth;
    }

    private void processGroundTruthBatch(Matrix ygroundTruth, ReplayBuffer buffer, RewardFunction rewardFunction,
                                         double discountFactor, Model networkToUse, List<Integer> stateIndices,
                                         List<Matrix> allInputs, List<Integer> moveCountsPerState)
    {
        try {
            int totalMoves = allInputs.size();
            int numCols = allInputs.get(0).getShape().getNumCols();
            Matrix batchedInput = Matrix.zeros(totalMoves, numCols);
            
            for (int i = 0; i < totalMoves; i++) {
                batchedInput.copySlice(i, i + 1, 0, numCols, allInputs.get(i));
            }

            Matrix batchedOutput = networkToUse.forward(batchedInput);

            int outputIdx = 0;
            for (int i = 0; i < stateIndices.size(); i++) {
                int rIdx = stateIndices.get(i);
                int numMoves = moveCountsPerState.get(i);
                
                Double maxQValue = null;
                for (int m = 0; m < numMoves; m++) {
                    double qVal = batchedOutput.get(outputIdx, 0);
                    if (maxQValue == null || qVal > maxQValue) {
                        maxQValue = qVal;
                    }
                    outputIdx++;
                }
                
                BattleView state = buffer.getState(rIdx);
                MoveView action = buffer.getAction(rIdx);
                BattleView nextState = buffer.getNextState(rIdx);
                double reward = rewardFunction.getStateActionStateReward(state, action, nextState);
                
                if (maxQValue != null) {
                    ygroundTruth.set(rIdx, 0, reward + discountFactor * maxQValue);
                } else {
                    ygroundTruth.set(rIdx, 0, reward);
                }
            }

        } catch (Exception e) {
            System.err.println("Warning: Ground truth batch processing failed, falling back to individual. " + e.getMessage());
            processGroundTruthIndividually(
                ygroundTruth, buffer, rewardFunction, discountFactor, networkToUse, stateIndices
            );
        }
    }

    private void processGroundTruthIndividually(Matrix ygroundTruth, ReplayBuffer buffer, RewardFunction rewardFunction,
                                                double discountFactor, Model networkToUse, List<Integer> stateIndices)
    {
        for (int rIdx : stateIndices) {
            BattleView state = buffer.getState(rIdx);
            MoveView action = buffer.getAction(rIdx);
            BattleView nextState = buffer.getNextState(rIdx);
            double reward = rewardFunction.getStateActionStateReward(state, action, nextState);
            
            List<MoveView> legalMoves = getAllLegalMoves(nextState);
            if (legalMoves.isEmpty()) {
                ygroundTruth.set(rIdx, 0, reward);
                continue;
            }

            Double maxQValue = null;
            for (MoveView move : legalMoves) {
                try {
                    Matrix input = this.getModelInput(nextState, move);
                    Matrix output = networkToUse.forward(input);
                    double qVal = output.get(0, 0);
                    if (maxQValue == null || qVal > maxQValue) {
                        maxQValue = qVal;
                    }
                } catch (Exception e) {
                }
            }
            
            if (maxQValue != null) {
                ygroundTruth.set(rIdx, 0, reward + discountFactor * maxQValue);
            } else {
                ygroundTruth.set(rIdx, 0, reward);
            }
        }
    }

    private MoveView argmaxFromList(BattleView view, List<MoveView> moves)
    {
        if (moves == null || moves.isEmpty()) {
            return null;
        }
        
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
                System.err.println("Error initializing model in argmaxFromList: " + e.getMessage());
                return moves.get(0);
            }
        }
        
        try {
            int numMoves = moves.size();
            List<Matrix> inputs = new ArrayList<>(numMoves);
            
            for (MoveView move : moves) {
                Matrix input = this.getModelInput(view, move);
                inputs.add(input);
            }
            
            int numCols = inputs.get(0).getShape().getNumCols();
            Matrix batchedInput = Matrix.zeros(numMoves, numCols);
            for (int i = 0; i < numMoves; i++) {
                batchedInput.copySlice(i, i + 1, 0, numCols, inputs.get(i));
            }
            
            Matrix batchedOutput = this.getModel().forward(batchedInput);
            
            Double maxVal = null;
            MoveView bestMove = null;
            for (int i = 0; i < numMoves; i++) {
                double qVal = batchedOutput.get(i, 0);
                if (maxVal == null || qVal > maxVal) {
                    maxVal = qVal;
                    bestMove = moves.get(i);
                }
            }
            
            return (bestMove != null) ? bestMove : moves.get(0);
            
        } catch (Exception e) {
            System.err.println("Warning: Batch inference failed in argmaxFromList, using first move: " + e.getMessage());
            return moves.get(0);
        }
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
        
        List<MoveView> legalMoves = getAllLegalMoves(view);
        return argmaxFromList(view, legalMoves);
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
        
        consecutiveSwitchChoices = 0;
        
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