package com.sngular.tutorial.mnist.dj4j;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nadam;

import java.io.File;

import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

@Slf4j
public class SimpleMNISTTest {

    private static DataSetIterator mnistTrain;
    private static DataSetIterator mnistTest;
    private static int seed = 8675309;
    private static int batchSize = 64;

    @BeforeAll
    @SneakyThrows
    public static void initAll() {
        //This is a convenience pre-built dataset included with the platform.
        mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        mnistTest = new MnistDataSetIterator(batchSize, false, seed);
    }


    @Test
    @SneakyThrows
    public void createSimpleModelLowLearningRate() {

        final int inputLayerRows = 28;
        final int inputLayerColumns = 28;
        int outputLayerClasses = 10;
        int epochCount = 5;
        double learningRate = 0.00001;
        DenseLayer inputLayer = new DenseLayer.Builder()
                .nIn(inputLayerRows * inputLayerColumns)
                .nOut(100)
                .build();

        OutputLayer outputLayer = new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(100)
                .nOut(outputLayerClasses)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) //randomizes values
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .l2(learningRate)
                .list()
                .layer(inputLayer)
                .layer(outputLayer)
                .build();

        final long startTime = System.currentTimeMillis();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
//        model.setListeners(new ScoreIterationListener(1));
        model.fit(mnistTrain, epochCount);
        final long endTime = System.currentTimeMillis();
        Evaluation evaluation = model.evaluate(mnistTest);
        log.info(evaluation.stats());
        log.info("Total Train Time:{}", endTime - startTime);
    }


    @Test
    @SneakyThrows
    public void createSimpleModelHigherLearningRate() {

        final int inputLayerRows = 28;
        final int inputLayerColumns = 28;
        int outputLayerClasses = 10;
        int epochCount = 5;
        double learningRate = .01;
        DenseLayer inputLayer = new DenseLayer.Builder()
                .nIn(inputLayerRows * inputLayerColumns)
                .nOut(100)
                .build();

        OutputLayer outputLayer = new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(100)
                .nOut(outputLayerClasses)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) //randomizes values
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .l2(learningRate)
                .list()
                .layer(inputLayer)
                .layer(outputLayer)
                .build();

        final long startTime = System.currentTimeMillis();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
//        model.setListeners(new ScoreIterationListener(1));
        model.fit(mnistTrain, epochCount);
        final long endTime = System.currentTimeMillis();
        Evaluation evaluation = model.evaluate(mnistTest);
        log.info(evaluation.stats());
        log.info("Total Train Time:{}", endTime - startTime);
    }


    @Test
    @SneakyThrows
    public void trainAndSaveModel() {

        final int inputLayerRows = 28;
        final int inputLayerColumns = 28;
        int outputLayerClasses = 10;
        int epochCount = 5;
        double learningRate = .01;
        DenseLayer inputLayer = new DenseLayer.Builder()
                .nIn(inputLayerRows * inputLayerColumns)
                .nOut(100)
                .build();

        OutputLayer outputLayer = new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(100)
                .nOut(outputLayerClasses)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) //randomizes values
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .l2(learningRate)
                .list()
                .layer(inputLayer)
                .layer(outputLayer)
                .build();

        final long startTime = System.currentTimeMillis();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
//        model.setListeners(new ScoreIterationListener(1));
        model.fit(mnistTrain, epochCount);
        final long endTime = System.currentTimeMillis();
        ModelSerializer.writeModel(model, "src/test/resources/mnistDJLModel.zip", false);
    }

    @Test
    @SneakyThrows
    public void loadModelAndEvaluate() {
        MultiLayerNetwork network = MultiLayerNetwork.load(new File("src/test/resources/mnistDJLModel.zip"), false);
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 123;
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
        Evaluation eval = network.evaluate(mnistTest);

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}
