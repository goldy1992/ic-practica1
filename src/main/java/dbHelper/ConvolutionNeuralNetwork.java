package dbHelper;

import com.google.common.base.Stopwatch;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

/**
 * Created by Mike on 04/12/2016.
 */
public class ConvolutionNeuralNetwork {

    private static final Logger log = LoggerFactory.getLogger(ConvolutionNeuralNetwork.class);

    public static void main(String[] args) throws Exception {

        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        int batchSize = 64; // Test batch size
        int nEpochs = 1; // Number of training epochs
        int iterations = 1; // Number of training iterations
        int seed = 123; //

        /*
            Create an iterator using the batch size for one iteration
         */
        log.info("Load data....");
     MnistData mnistData = new MnistData(batchSize, seed);
        /*
            Construct the neural network
         */
        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations) // Training iterations as above
                .regularization(true).l2(0.0005)
                /*
                    Uncomment the following for learning decay and bias
                 */
                .learningRate(.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false);

                        /*
-        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
-        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
-            and the dense layers
-        (b) Does some additional configuration validation
-        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
-            layer based on the size of the previous layer (but it won't override values manually set by the user)
-
-        In earlier versions of DL4J, the (now deprecated) ConvolutionLayerSetup class was used instead for this.
-        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
-        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
-        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
-        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
-         */
        // The builder needs the dimensions of the image along with the number of channels. these are 28x28 images in one channel

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        Stopwatch timer = new Stopwatch().start();
        model.init();


        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(mnistData.getMnistTrain());
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(mnistData.getMnistTest().hasNext()){
                DataSet ds = mnistData.getMnistTest().next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);

            }
            log.info(eval.stats());
            mnistData.getMnistTest().reset();
        }
        log.info("****************Example finished********************");
        timer.stop();
        long totalSeconds = timer.elapsedTime(TimeUnit.SECONDS);
        long excessSeconds = totalSeconds % 60;
        long minutes = (totalSeconds - excessSeconds) / 60;
        log.info("Total time: " + (minutes < 10 ? "0" + minutes : minutes) + ":" + (excessSeconds < 10 ? "0" + excessSeconds : excessSeconds));
    }
}
