package ece.cpen502;


import robocode.RobocodeFileOutputStream;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNet implements  NeuralNetInterface {

    private int numInput;
    private int numHidden;
    private int numOutput;
    private double  learningRate;
    private double momentum;
    private double a;
    private double b;
    private boolean binary;

    //training set
    private double[][] inputs; // {{0,0},{0,1},{1,0},{1,1}};
    private double[][] expectedOutputs; // {{0},{1},{1},{0}};
    private double[][] outputs;

    private double[] inputLayer;
    private double[] hiddenLayer;
    private double[] outputLayer;


    private double[][] weightsHidden; //input-to-hidden weights
    private double[][]weightsOutput; //hidden-to-output weights

    private double[][] deltaWeightsHidden; //previous weight change of weightsHidden
    private double[][] deltaWeightsOutput; //previous weight change of weightsOutput

    private double[] deltaHidden;
    private double[] deltaOutput;

    private List<Double> epochErrors;

//    private int numStates;
    private int numActions;
    double[] factors = {4,3,4,4,4,6};

//    private int state;
    private double maxValue = Double.MIN_VALUE;
    private int maxAction;
    private double currentQ;
    private double nextQ;
    private double[] trainInput;
    private double[] trainExpectedOutput;

    private boolean initialStateAction = true;

    private double epsilon = 0.25;
    private double gamma = 0.9; // [0.1,0.3,0.5,0.7,0.9]
    private double alpha = 0.1;

    private double upperbound = 0.5;
    private double lowerbound = -0.5;

    public NeuralNet(int numInput,
                        int numHidden,
                        int numOutput,
                        double learningRate,
                        double momentum,
                        boolean binary) {
//        this.inputs = inputs;
//        this.expectedOutputs = expectedOutputs;
//        this.numInputs = inputs[0].length;
        this.numInput = numInput;
        this.numHidden = numHidden;
//        this.numOutputs = expectedOutputs[0].length;
        this.numOutput = numOutput;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.binary = binary;
        if(binary) {
            this.a = 0;
            this.b = 1;
        } else {
            this.a = -1;
            this.b = 1;
        }
        this.setting();
    }

    public void setInputs(double[][] inputs) {
        this.inputs = normalizeInputs(inputs);
        outputs = new double[inputs.length][numOutput];
        numActions = inputs.length;
    }

    public double[][] normalizeInputs(double[][] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                inputs[i][j] /= (factors[j] - 1); // 0123 -> 0-1
            }

        }
        return inputs;
    }

    public void setExpectedOutputs(double[][] expectedOutputs) {
        this.expectedOutputs = expectedOutputs;
    }

    public int epsilonGreedy() {
        getMaxAction();
        double random = Math.random();
        int nextAction;
        if (random < epsilon) {
            double randomAction = Math.random();
            nextAction = (int)(numActions * randomAction);
        } else {
            nextAction = maxAction;
        }
        nextQ = outputs[nextAction][0]; // next state + chosen action -> next Q-value
        return nextAction;
    }

    public int getMaxAction () {
        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = forwardPropagation(inputs[i]);
            if (outputs[i][0] > maxValue) {
                maxValue = outputs[i][0];
                maxAction = i;
            }
        }
        return maxAction;
    }


    public void QLearning (double[] input, double reward, boolean online) {
        if (initialStateAction) {
            initialStateAction = false;
        } else {
            currentQ = currentQ + alpha * (reward + gamma * maxValue - currentQ); //update current Q-value
            trainExpectedOutput = new double[] {currentQ};
            forwardPropagation(trainInput);
            if (online) {
                backforwardPropagation(trainExpectedOutput); // train (input, Q)
            }
        }
        currentQ = nextQ; // update next curretQ
        trainInput = input; // update next training input
    }


    public void sarsa (double[] input, double reward, boolean online) {
        if (initialStateAction) {
            initialStateAction = false;
        } else {
            currentQ  = currentQ + alpha * (reward + gamma * nextQ - currentQ); // update current Q-value
            trainExpectedOutput = new double[] {currentQ};
            forwardPropagation(trainInput);
            if (online) {
                backforwardPropagation(trainExpectedOutput);
            }

        }
        currentQ = nextQ;
        trainInput = input;
    }


    @Override
    public double sigmoid(double x) {
        return 2 / (1 + Math.exp(-x)) - 1;
    }

    @Override
    public double customSigmoid(double x) {
        return (b - a) / (1 + Math.exp(-x)) + a;
    }

    public double customSigmoidDerivative(double x) {
        return -(1.0/(b-a)) * (x-a) * (x-b);
    }

    @Override
    public void initializeWeights() {
        for (int i = 0; i < weightsHidden.length; i++) {
            for (int j = 0; j < weightsHidden[0].length; j++) {
                double random = new Random().nextDouble();
                weightsHidden[i][j] = lowerbound + (upperbound - lowerbound)*random;
            }
        }

        for (int i = 0; i < weightsOutput.length; i++) {
            for (int j = 0; j < weightsOutput[0].length; j++) {
                double random = new Random().nextDouble();
                weightsOutput[i][j] = lowerbound + (upperbound - lowerbound)*random;
            }
        }
    }


    @Override
    public void zeroWeights() {
        for (int i = 0; i < weightsHidden.length; i++) {
            for (int j = 0; j < weightsHidden[0].length; j++) {
                weightsHidden[i][j] = 0;
            }
        }
        for (int i = 0; i < weightsOutput.length; i++) {
            for (int j = 0; j < weightsOutput[0].length; j++) {
                weightsOutput[i][j] = 0;
            }
        }
    }


    public void setting() {

        // initialize layers
        inputLayer= new double[numInput + 1]; // one more neuron for bia
        hiddenLayer = new double[numHidden + 1]; // one more neuron for bia
        outputLayer = new double[numOutput];

//        outputs = new double[inputs.length][numOutputs];

        weightsHidden = new double[numHidden][numInput + 1];
        weightsOutput = new double[numOutput][numHidden + 1];

        deltaWeightsHidden = new double[numHidden][numInput + 1];
        deltaWeightsOutput = new double[numOutput][numHidden + 1];

        deltaHidden = new double[numHidden];
        deltaOutput = new double[numOutput];

        initializeWeights();
    }

    public void setInput(double[] X) {

        //set up inputLayer and hiddenLayer with bia
        inputLayer[X.length] = 1;
        for (int i = 0; i < X.length; i++) {
            inputLayer[i] = X[i];
        }
        hiddenLayer[numHidden] = 1;
        for (int i = 0; i < numHidden; i++) {
            hiddenLayer[i] = 0;
        }
    }

    @Override
    public double outputFor(double[] X) {
        return 0.0;
    }

    public double[] forwardPropagation(double[] X) {

        setInput(X);

        //calculate hiddensLayer
        for (int i = 0; i < weightsHidden.length; i++) {
            hiddenLayer[i] = 0;
            for (int j = 0; j < weightsHidden[0].length; j++) {
                hiddenLayer[i] += weightsHidden[i][j] * inputLayer[j];
            }
            hiddenLayer[i] = customSigmoid(hiddenLayer[i]);
        }

        //calculate outputLayer
        for (int i = 0; i < weightsOutput.length; i++) {
            outputLayer[i] = 0;
            for (int j = 0; j < weightsOutput[0].length; j++) {
                outputLayer[i] += weightsOutput[i][j] * hiddenLayer[j];
            }
            outputLayer[i] = customSigmoid(outputLayer[i]);
        }
        return outputLayer;
    }


    public void backforwardPropagation(double[] expectedOutput) {

        // update weightOutput
        // delta_j = derivative(yi)(cj - yj)
        // wji_new = wji + p * delta_j * xi + a * delta_wji
        for (int j = 0; j < weightsOutput.length; j++) {
            double yj = outputLayer[j];
            double cj = expectedOutput[j];
            deltaOutput[j] = customSigmoidDerivative(yj) * (cj - yj);
            deltaOutput[j] = cj - yj;
            for (int i = 0; i < weightsOutput[0].length; i++) {
                deltaWeightsOutput[j][i] = momentum * deltaWeightsOutput[j][i] + learningRate * deltaOutput[j] * hiddenLayer[i];
                weightsOutput[j][i] += deltaWeightsOutput[j][i];
            }
        }

        // update weightHidden
        // delta_j = derivative(yi) * sum(delta_h * whj)
        // wji_new = wji + p * delta_j * xi + a * delta_wji

        for (int j = 0; j < weightsHidden.length; j++) {
            double sum  = 0;
            for (int h = 0; h < outputLayer.length; h++) {
                double delta_h = deltaOutput[h];
                sum += delta_h * weightsOutput[h][j];
            }
            double yj = hiddenLayer[j];
            deltaHidden[j] = customSigmoidDerivative(yj) * sum;

            for (int i = 0; i < weightsHidden[0].length; i++) {
                deltaWeightsHidden[j][i] = momentum * deltaWeightsHidden[j][i] + learningRate * deltaHidden[j] * inputLayer[i];
                weightsHidden[j][i] += deltaWeightsHidden[j][i];
            }
        }

    }

    @Override
    public double train(double[] X, double argValue) {
        return 0.0;
    }

    /**
     * training all input data
     */
    public double trainAll() {
        double totalError = 0;
        for (int i = 0; i < inputs.length; i++) {
            double error = 0;
            double[] output = forwardPropagation(inputs[i]);
            outputs[i] = output;
            for (int j = 0; j < numOutput; j++) {
                error += Math.pow(output[j] - expectedOutputs[i][j], 2) / 2; // (y - y')^2/2
            }
            totalError = totalError + error;
            backforwardPropagation(expectedOutputs[i]);
        }
        epochErrors.add(totalError);
        return totalError;
    }

    public double[][] getOutputs() {
        return outputs;
    }

    public double predict() {
        double testError = 0;
        for (int i = 0; i < inputs.length; i++) {
            double error = 0;
            double[] output = forwardPropagation(inputs[i]);
            outputs[i] = output;
            for (int j = 0; j < numOutput; j++) {
                error += Math.pow(output[j] - expectedOutputs[i][j], 2) / 2; // (y - y')^2/2
            }
            testError += error;
        }
        return Math.sqrt(2 * testError / inputs.length); // RMS error
    }

    /**
     * run multiple epochs to reach a total error less than target error
     *
     * @param maxNumEpo
     * @return epochError at maxNumEpo
     */
    public double converge(int maxNumEpo) {
        int epoch = 0;
        epochErrors = new ArrayList<>();
        double epochError = 0;
        while (epoch < maxNumEpo) {
            epochError = Math.sqrt(2 * trainAll() / inputs.length); // RMS error
            epoch++;
        }
        return epochError;
    }

    public void load(File argFile){

        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(argFile));

            int i = 0;
            boolean one = true;
            String line;
            while ((line = reader.readLine()) != null && one) {
                if (line.equals("*")) {
                    one = false;
                    break;
                }
                String[] cols = line.split(",");
                for (int j = 0; j < cols.length; j++) {
                    weightsHidden[i][j] = Double.valueOf(cols[j]);
                }
                i++;
            }
            i = 0;
            while ((line = reader.readLine()) != null && !one) {
                String[] cols = line.split(",");
                for (int j = 0; j < cols.length; j++) {
                    weightsOutput[i][j] = Double.valueOf(cols[j]);
                }
                i++;
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


    @Override
    public void save(File argFile){

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < weightsHidden.length; i++) {
            for (int j = 0; j < weightsHidden[0].length; j++) {
                sb.append(weightsHidden[i][j] + "");
                if (j < weightsHidden[0].length - 1) {
                    sb.append(",");
                }
            }
            sb.append("\n");
        }
        sb.append("*\n");
        for (int i = 0; i < weightsOutput.length; i++) {
            for (int j = 0; j < weightsOutput[0].length; j++) {
                sb.append(weightsOutput[i][j] + "");
                if (j < weightsOutput[0].length - 1) {
                    sb.append(",");
                }
            }
            sb.append("\n");
        }

        try {
            PrintStream ps = new PrintStream(new RobocodeFileOutputStream(argFile));
            ps.println(sb.toString());
            ps.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}