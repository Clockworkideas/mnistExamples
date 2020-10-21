package com.sngular.tutorial.mnist.djl;

import ai.djl.Model;
import ai.djl.basicdataset.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.CheckpointsTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import lombok.SneakyThrows;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class SimpleMnist {


    private static final String MODEL_DIR = "target/models/mnist";
    private static final String MODEL_NAME = "mlp";
    private static RandomAccessDataset trainingSet;
    private static RandomAccessDataset validateSet;

    @BeforeAll
    @SneakyThrows
    public static void initClass() {
        trainingSet = prepareDataset(Dataset.Usage.TRAIN, 64, Long.MAX_VALUE);
        validateSet = prepareDataset(Dataset.Usage.TEST, 64, Long.MAX_VALUE);
    }

    @SneakyThrows
    private static RandomAccessDataset prepareDataset(Dataset.Usage usage, int batchSize, long limit) {
        Mnist mnist = Mnist.builder().optUsage(usage).setSampling(batchSize, true).optLimit(limit).build();
        mnist.prepare(new ProgressBar());
        return mnist;
    }

    @Test
    @SneakyThrows
    public void testTrainAndEval() {
        final int inputLayerRows = 28;
        final int inputLayerColumns = 28;
        int outputLayerClasses = 10;
        Block mlpBlock = new Mlp(inputLayerRows * inputLayerColumns, outputLayerClasses, new int[]{128});
        Model model = Model.newInstance(MODEL_NAME);
        model.setBlock(mlpBlock);
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging())
                .addTrainingListeners(new CheckpointsTrainingListener(MODEL_DIR));

        Trainer trainer = model.newTrainer(config);
        trainer.setMetrics(new Metrics());
        Shape inputShape = new Shape(1, Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH);
        trainer.initialize(inputShape);
        EasyTrain.fit(trainer, 10, trainingSet, validateSet);
    }

    @Test
    @SneakyThrows
    public void testWithModel() {
        //test with model
        var img = ImageFactory.getInstance().fromUrl("src/test/resources/exampleSet/5/img_8.jpg");
//        var img = ImageFactory.getInstance().fromUrl("src/test/resources/exampleSet/misc/random1.jpg");
        img.getWrappedImage();
        Path modelDir = Paths.get("models/mlp");
        Model model = Model.newInstance("mlp");
        model.setBlock(new Mlp(28 * 28, 10, new int[]{128, 64}));
        model.load(modelDir);

        Translator<Image, Classifications> translator = new Translator<Image, Classifications>() {

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                // Convert Image to NDArray
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
                return new NDList(NDImageUtils.toTensor(array));
            }

            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                // Create a Classifications with the output probabilities
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }

            @Override
            public Batchifier getBatchifier() {
                // The Batchifier describes how to combine a batch together
                // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
                return Batchifier.STACK;
            }
        };

        var predictor = model.newPredictor(translator);
        var classifications = predictor.predict(img);
        System.out.println(classifications);
    }
}
