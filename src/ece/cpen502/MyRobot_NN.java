package ece.cpen502;

import robocode.*;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import static robocode.util.Utils.normalRelativeAngleDegrees;

public class MyRobot_NN extends AdvancedRobot {


    // Neural network : state-action -> neural network -> Q value

    private int numRounds = 10000;
    private boolean terminalRewardOnly = false;
    private double reward = 0.0;
    private double cumulativeReward = 0.0;
    private static List<Double> cumulativeRewards = new ArrayList<>();


    private boolean offPolicy = true;

    private int nextAction;
//    private int nextState;

    private double enemyDistance;
    private double enemyBearing;


    // quatization factors
    private int numX = 4;
    private int numY = 3;
    private int numHeading = 4;
    private int numEnemyDistance = 4;
    private int numEnemyBearing = 4;
    private int numState = numX * numY * numHeading * numEnemyDistance * numEnemyBearing;
    private int numAction = 7;

    private int numInput = 6; // input = [X, Y, Heading, EnemyDistance, EnemyBearing, Action]
    private int numHidden = 5; // chosen
    private int numOutput = 1; // output = [Q-value]
    private double learningRate = 0.0001; // chosen
    private double momentum = 0.9; // chosen
    private boolean binary = true;


    private int quantizedX;
    private int quantizedY;
    private int quantizedHeading;
    private int quantizedEnemyDistance;
    private int quantizedEnemyBearing;

    private double[][] inputs = new double[numAction][numInput];
//    private double[][] expectedOutputs =  new double[numAction][numOutput];

//    private LUT lut = new LUT(numState, numAction);
    private NeuralNet nn = new NeuralNet(numInput, numHidden, numOutput, learningRate, momentum, binary);

    private static List<Integer> battleResults = new ArrayList<>();

    public void run() {

        if (getRoundNum() > 0) {
//            lut.load(getDataFile("LUT.txt"));
            nn.load(getDataFile("weights.txt")); // load updated weights
        }

        setColors(Color.BLACK,Color.YELLOW,Color.BLACK);

        // Loop forever
        while (true) {

            getQuatizedState();
//            nextAction = lut.epsilonGreedy(nextState);

            // inputs = nextState + action1, nextState + action2, ...
            for (int i = 0; i < inputs.length; i++) {
                inputs[i] = new double[]{quantizedX, quantizedY, quantizedHeading, quantizedEnemyDistance, quantizedEnemyBearing, i};
            }

            nn.setInputs(inputs);
            nextAction = nn.epsilonGreedy(); //randomly choose or get maxAction

            if (offPolicy) {
//                lut.QLearning(nextState, nextAction, reward);
                nn.QLearning(inputs[nextAction], reward, false);
            } else {
//                lut.sarsa(nextState, nextAction, reward);
                nn.sarsa(inputs[nextAction], reward, false);
            }
//            nn.save(new File("weights.txt"));
            cumulativeReward += reward;
            reward = 0.0; // reset reward before taking next action

            takeAction(nextAction);
            //scan
            turnRadarLeft(360);
        }
    }

    /**
     Quantize a state
     */
    public int getQuatizedState() {

        quantizedX = (int) (getX() / (getBattleFieldWidth() / numX));
        quantizedY = (int) (getY() / (getBattleFieldHeight() / numY));
        quantizedHeading = (int) (getHeading() / (360 / numHeading)); // quantize heading [0, 360) to 0 ~ 3

        double diagonal  = Math.sqrt(getBattleFieldWidth() * getBattleFieldWidth() + getBattleFieldHeight() * getBattleFieldHeight());
        quantizedEnemyDistance = (int) (enemyDistance / (diagonal / numEnemyDistance));

        double bearing = enemyBearing;
        if (enemyBearing < 0) {
            bearing = 360 + enemyBearing; // enemyBearing range [-180, 180)
        }
        quantizedEnemyBearing = (int) (bearing / (360 / numEnemyBearing));

        int prod = numState / numX;
        int index = quantizedX * prod;
        prod = prod / numY;
        index += quantizedY * prod;
        prod = prod / numHeading;
        index += quantizedHeading * prod;
        prod = prod / numEnemyDistance;
        index += quantizedEnemyDistance * prod;
        prod = prod / numEnemyBearing;
        index += quantizedEnemyBearing * prod;

        return index;

    }

    public void takeAction(int action) {
        switch (action) {
            case 0: // move ahead
                setAhead(100);
                break;
            case 1: // move back
                setBack(100);
                break;
            case 2: // turn left and move ahead
                setTurnLeft(45);
                setAhead(100);
                break;
            case 3: // turn right and move ahead
                setTurnRight(45);
                setAhead(100);
                break;
            case 4: // turn left and move back
                setTurnLeft(45);
                setBack(100);
                break;
            case 5: // turn right and move back
                setTurnRight(45);
                setBack(100);
                break;
            case 6:
                if (quantizedEnemyDistance == 2) {
                    fire(1);
                }
                else if (quantizedEnemyDistance == 1) {
                    fire(2);
                }
                else if (quantizedEnemyDistance == 0) {
                    fire(3);
                }
        }
        execute();
    }

    /**
     * onScannedRobot:  return parameters of the enemy
     */
    public void onScannedRobot(ScannedRobotEvent enemy) {

        enemyBearing = enemy.getBearing(); //relative to your robot's heading, in degrees (-180 <= getBearing() < 180)
        enemyDistance = enemy.getDistance();

        double gunTurnAmt = normalRelativeAngleDegrees(enemyBearing + getHeading() - getGunHeading());
        setTurnGunRight(gunTurnAmt);

//        if (quantizedEnemyDistance == 2) {
//            fire(1);
//        }
//        else if (quantizedEnemyDistance == 1) {
//            fire(2);
//        }
//        else if (quantizedEnemyDistance == 0) {
//            fire(3);
//        }
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        double interReward =  event.getBullet().getPower() * 9;
        if (!terminalRewardOnly) {
            reward += interReward;
        }
    }

    @Override
    public void onBulletMissed(BulletMissedEvent event) {
        double interReward = -event.getBullet().getPower();
        if (!terminalRewardOnly) {
            reward += interReward;
        }
    }

    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        double bulletPower = event.getPower();
        double interReward =  - 6 * bulletPower;
        if (!terminalRewardOnly) {
            reward += interReward;
        }
    }

    @Override
    public void onHitRobot(HitRobotEvent event) {
        //double bulletPower = event.getPower();
        double interReward = -6;
        if (!terminalRewardOnly) {
            reward += interReward;
        }
    }



    @Override
    public void onHitWall(HitWallEvent event) {
        double interReward = -(Math.abs(getVelocity()) * 0.5 - 1);
        if (!terminalRewardOnly) {
            reward += interReward;
        }
    }


    @Override
    public void onWin(WinEvent event) {
        reward += 20;
        battleResults.add(1);
        cumulativeRewards.add(cumulativeReward);
    }

    @Override
    public void onDeath(DeathEvent event) {
        reward -= 20;
        battleResults.add(0);
        cumulativeRewards.add(cumulativeReward);

    }

    @Override
    public void onRoundEnded(RoundEndedEvent event) {
//        lut.save(getDataFile("LUT.txt"));
        nn.save(getDataFile("weights.txt"));
    }


    @Override
    public void onBattleEnded(BattleEndedEvent event) {
        saveResults(getDataFile("battle_results.txt"));
        saveRewards(getDataFile("rewards.txt"));
    }


    public void saveResults(File argFile) {
        PrintStream s = null;
        try   {
            s = new PrintStream(new RobocodeFileOutputStream(argFile));
            for (int i = 0; i < battleResults.size(); i++) {
                    s.println(new Integer(battleResults.get(i)));
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            s.close();
        }
    }

    public void saveRewards(File argFile) {
        PrintStream s = null;
        try   {
            s = new PrintStream(new RobocodeFileOutputStream(argFile));
            for (int i = 0; i <cumulativeRewards.size(); i++) {
                s.println(new Double(cumulativeRewards.get(i)));
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            s.close();
        }
    }
}
