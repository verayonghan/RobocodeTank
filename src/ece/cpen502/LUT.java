package ece.cpen502;

import robocode.RobocodeFileOutputStream;

import java.io.*;

public class LUT implements LUTInterface{

    private double[][] table; //look up table[state][action]

    private int numStates;
    private int numActions;

    private int state;
    private int action;

    private boolean initialStateAction = true;

    private double epsilon = 0.5;
    private double gamma = 0.9;
    private double alpha = 0.3;

    public LUT(int numStates, int numActions) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.table = new double[numStates][numActions];
        initialiseLUT();
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setGamma(double gamma) {
        this.gamma = gamma;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public void initialiseLUT() {
        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < numActions; j++) {
                table[i][j] = 0;
            }
        }
    }

    public double getValue(int state, int action) {
        return table[state][action];
    }


    /**
     * off-policy
     *
     * Q(s,a) = Q(s,a) + alpha[reward + gamma * max_Q(s',a') - Q(s,a)]
     *
     * @param nextState
     * @param nextAction
     * @param reward
     */

    public void QLearning (int nextState, int nextAction, double reward) {
//        int nextAction = epsilonGreedy(nextState);
        if (initialStateAction) {
            initialStateAction = false;
        } else {
            int maxAction = getMaxAction(nextState);
            table[state][action] = table[state][action] + alpha * (reward + gamma * table[nextState][maxAction] - table[state][action]);
        }
        state = nextState;
        action = nextAction;
    }

    /**
     * on-policy
     *
     * @param nextState
     * @param nextAction
     * @param reward
     */
    public void sarsa (int nextState, int nextAction, double reward) {
//        int nextAction = epsilonGreedy(nextState);
        if (initialStateAction) {
            initialStateAction = false;
        } else {
            table[state][action] = table[state][action] + alpha * (reward + gamma * table[nextState][nextAction] - table[state][action]);
        }
        state = nextState;
        action = nextAction;
    }

    /**
     * select random action with epsilon probability
     * select an action with (1-epsilon) probability that gives maximum reward in given state
     *
     * @param state
     * @return
     */

    public int epsilonGreedy(int state) {
        double random = Math.random();
        int action;
        if (random < epsilon) {
            double randomAction = Math.random();
            action = (int)(numActions * randomAction);
        } else {
            action = getMaxAction(state);
        }

        return action;
    }

    public int getMaxAction (int state) {
        double maxValue = Double.MIN_VALUE;
        int maxAction = 0;
        for (int i = 0; i < numActions; i++) {
            if (table[state][i] > maxValue) {
                maxValue = table[state][i];
                maxAction = i;
            }
        }
        return maxAction;
    }


    @Override
    public int indexFor(double[] X) {
        return 0;
    }

    @Override
    public double outputFor(double[] X) {
        return 0.0;
    }

    @Override
    public double train(double[] X, double argValue) {
        return 0;
    }


    @Override
    public void save(File argFile) {

        try   {
            PrintStream ps  = new PrintStream(new RobocodeFileOutputStream(argFile));
            for (int i = 0; i < numStates; i++) {
                for (int j = 0; j < numActions; j++) {
                    ps.println(new Double(table[i][j]));
                }
            }

            ps.close();
        }
        catch (IOException e)   {
            e.printStackTrace();
        }
    }



    public void load(File argFile) {

        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(argFile));
            for (int i = 0; i < numStates; i++) {
                for (int j = 0; j < numActions; j++) {
                    table[i][j] = Double.valueOf(reader.readLine());
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                reader.close();
            } catch (IOException e) {
                e.printStackTrace();

            }
        }
    }

    @Override
    public void load(String argFileName) throws IOException {

    }
}
