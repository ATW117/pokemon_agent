package src.pas.pokemon.agents;


// SYSTEM IMPORTS
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import net.sourceforge.argparse4j.inf.Namespace;

import edu.bu.pas.pokemon.nn.Model;

/**
 * A PolicyAgent variant that loads one of the most recent checkpoints (based on
 * file modification time) for self-play. It can optionally consult a metrics log
 * (containing "after cycle=X ... avg(num_wins)=Y" entries) to bias sampling
 * toward historically stronger models.
 *
 * Configuration knobs (system properties or environment variables):
 *  - selfplay.dir / SELFPLAY_DIR: directory containing checkpoint .model files (default: params)
 *  - selfplay.metrics / SELFPLAY_METRICS: path to log file with cycle stats (default: logs/training.log)
 *  - selfplay.k / SELFPLAY_K: number of most recent checkpoints to consider (default: 20)
 */
public class SelfPlayPolicyAgent extends PolicyAgent
{
    private static final Pattern MODEL_CYCLE_PATTERN = Pattern.compile("(\\d+)(?=\\.model$)");
    private static final Pattern METRIC_PATTERN = Pattern.compile("after cycle=(\\d+) .*avg\\(num_wins\\)=(\\S+)");
    private static final double MIN_WEIGHT = 0.1;

    private final Random historyRng = new Random();

    @Override
    public void initialize(Namespace args)
    {
        super.initialize(args);
        if (!loadHistoricalModel()) {
            System.err.println("[SELFPLAY] No historical model loaded; using freshly initialized weights.");
        }
    }

    private boolean loadHistoricalModel() {
        File[] recentModels = selectRecentModels();
        if (recentModels == null || recentModels.length == 0) {
            System.err.println("[SELFPLAY] No checkpoint files found in directory: " + getModelDir());
            return false;
        }

        String metricsPath = getConfig("selfplay.metrics", "SELFPLAY_METRICS", "logs/training.log");
        Map<Integer, Double> winRates = loadWinRates(metricsPath);
        ModelCandidate candidate = pickCandidate(recentModels, winRates);
        if (candidate == null) {
            System.err.println("[SELFPLAY] Failed to pick a checkpoint candidate.");
            return false;
        }
        return loadModelFromFile(candidate);
    }

    private File[] selectRecentModels() {
        File dir = new File(getModelDir());
        if (!dir.exists() || !dir.isDirectory()) {
            return null;
        }
        File[] models = dir.listFiles((d, name) -> name != null && name.endsWith(".model"));
        if (models == null || models.length == 0) {
            return null;
        }
        Arrays.sort(models, Comparator.comparingLong(File::lastModified).reversed());
        int limit = Math.min(models.length, getSampleSize());
        return Arrays.copyOf(models, limit);
    }

    private ModelCandidate pickCandidate(File[] candidates, Map<Integer, Double> winRates) {
        List<ModelCandidate> weighted = new ArrayList<>();
        for (File file : candidates) {
            int cycle = extractCycleNumber(file.getName());
            double score = winRates.getOrDefault(cycle, 0.0);
            weighted.add(new ModelCandidate(file, score, cycle));
        }
        double totalWeight = 0.0;
        double[] weights = new double[weighted.size()];
        for (int i = 0; i < weighted.size(); i++) {
            double w = Math.max(MIN_WEIGHT, weighted.get(i).score);
            weights[i] = w;
            totalWeight += w;
        }
        if (totalWeight <= 0) {
            return weighted.isEmpty() ? null : weighted.get(0);
        }
        double pick = historyRng.nextDouble() * totalWeight;
        double running = 0.0;
        for (int i = 0; i < weighted.size(); i++) {
            running += weights[i];
            if (running >= pick) {
                return weighted.get(i);
            }
        }
        return weighted.get(weighted.size() - 1);
    }

    private boolean loadModelFromFile(ModelCandidate candidate) {
        try {
            Model model = this.getModel();
            if (model == null) {
                model = this.initModel();
                this.setModel(model);
            }
            model.load(candidate.file.getAbsolutePath());
            System.err.println(String.format(
                "[SELFPLAY] Loaded checkpoint %s (cycle=%d, avgWins=%.3f)",
                candidate.file.getName(),
                candidate.cycle,
                candidate.score));
            return true;
        } catch (Exception e) {
            System.err.println("[SELFPLAY] Failed to load checkpoint '" + candidate.file.getName() + "': " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    private Map<Integer, Double> loadWinRates(String metricsPath) {
        Map<Integer, Double> map = new HashMap<>();
        if (metricsPath == null || metricsPath.isEmpty()) {
            return map;
        }
        File logFile = new File(metricsPath);
        if (!logFile.exists()) {
            return map;
        }
        try (BufferedReader reader = new BufferedReader(new FileReader(logFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                Matcher matcher = METRIC_PATTERN.matcher(line);
                if (matcher.find()) {
                    int cycle = Integer.parseInt(matcher.group(1));
                    double wins = Double.parseDouble(matcher.group(2));
                    map.put(cycle, wins);
                }
            }
        } catch (IOException e) {
            System.err.println("[SELFPLAY] Unable to read metrics log '" + metricsPath + "': " + e.getMessage());
        }
        return map;
    }

    private int extractCycleNumber(String filename) {
        Matcher matcher = MODEL_CYCLE_PATTERN.matcher(filename);
        if (matcher.find()) {
            return Integer.parseInt(matcher.group(1));
        }
        return -1;
    }

    private String getModelDir() {
        return getConfig("selfplay.dir", "SELFPLAY_DIR", "params");
    }

    private int getSampleSize() {
        String prop = System.getProperty("selfplay.k");
        if (prop != null) {
            try {
                return Math.max(1, Integer.parseInt(prop));
            } catch (NumberFormatException ignored) {}
        }
        String env = System.getenv("SELFPLAY_K");
        if (env != null) {
            try {
                return Math.max(1, Integer.parseInt(env));
            } catch (NumberFormatException ignored) {}
        }
        return 20; // default window
    }

    private String getConfig(String propertyKey, String envKey, String defaultValue) {
        String value = System.getProperty(propertyKey);
        if (value != null && !value.isEmpty()) {
            return value;
        }
        value = System.getenv(envKey);
        if (value != null && !value.isEmpty()) {
            return value;
        }
        return defaultValue;
    }

    private static final class ModelCandidate {
        final File file;
        final double score;
        final int cycle;

        ModelCandidate(File file, double score, int cycle) {
            this.file = file;
            this.score = score;
            this.cycle = cycle;
        }
    }
}

