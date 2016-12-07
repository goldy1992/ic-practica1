package dbHelper;

import com.google.common.base.Stopwatch;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

/**
 * MNIST database utilities.
 * 
 * @author Fernando Berzal (berzal@acm.org)
 */
public class ForwardFeedNeuralNetwork
{
	// MNIST URL
	private static final String MNIST_URL = "http://yann.lecun.com/exdb/mnist/";
	
	// Training data
	private static final String trainingImages = "train-images-idx3-ubyte.gz";
	private static final String trainingLabels = "train-labels-idx1-ubyte.gz";
	
	// Test data
	private static final String testImages = "t10k-images-idx3-ubyte.gz";
	private static final String testLabels = "t10k-labels-idx1-ubyte.gz";
	
	// Logger
	protected static final Logger log = Logger.getLogger(ForwardFeedNeuralNetwork.class.getName());
	
	// Download files
	
	/**
	 * Download URL to file using Java NIO.
	 * 
	 * @param url Source URL
	 * @param filename Destination file name
	 */
	public static void download (String urlString, String filename)
		throws IOException
	{
		URL url = new URL(urlString);
		ReadableByteChannel rbc = Channels.newChannel(url.openStream());
		FileOutputStream fos = new FileOutputStream(filename);
		fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
		fos.close();
		rbc.close();
	}

	/**
	 * Download MNIST database.
	 * 
	 * @param directory Destination folder/directory.
	 * @throws IOException
	 */
	public static void downloadMNIST (String directory) throws IOException 
	{
		File baseDir = new File(directory);
		
		if (!(baseDir.isDirectory() || baseDir.mkdir())) {
			throw new IOException("Unable to create destination folder " + baseDir);
		}
		
		log.info("Downloading MNIST database...");
		
		download(MNIST_URL+trainingImages, directory+trainingImages);
		download(MNIST_URL+trainingLabels, directory+trainingLabels);
		download(MNIST_URL+testImages, directory+testImages);
		download(MNIST_URL+testLabels, directory+testLabels);

		log.info("MNIST database downloaded into "+directory);
	}

	
	// Read data from files
	
	/**
	 * Read MNIST image data.
	 * 
	 * @param filename File name
	 * @return 3D int array
	 * @throws IOException
	 */
	public static int[][][] readImages (String filename)
			throws IOException
	{
		FileInputStream file = null;
		InputStream gzip = null;
		DataInputStream data = null;
		int images[][][] = null;

		try {
//			file = new FileInputStream(filename);
			ForwardFeedNeuralNetwork mnistDatabase = new ForwardFeedNeuralNetwork();
			ClassLoader classLoader = mnistDatabase.getClass().getClassLoader();
			file = new FileInputStream(classLoader.getResource(filename).getFile());
			gzip = new GZIPInputStream(file);
			data = new DataInputStream(gzip);

			log.info("Reading MNIST data...");

			int magicNumber = data.readInt();

			if (magicNumber!=2051) // 0x00000801 == 08 (unsigned byte) + 03 (3D tensor, i.e. multiple 2D images)
				throw new IOException("Error while reading MNIST data from "+filename);

			int size = data.readInt();
			int rows = data.readInt();
			int columns = data.readInt();

			images = new int[size][rows][columns];
			
			log.info("Reading "+size+" "+rows+"x"+columns+" images...");

			for (int i=0; i<size; i++)
				for (int j=0; j<rows; j++)
					for (int k=0; k<columns; k++)
						images[i][j][k] = data.readUnsignedByte();

			log.info("MNIST images read from "+filename);

		} finally {

			if (data!=null)
				data.close();
			if (gzip!=null)
				gzip.close();
			if (file!=null)
				file.close();
		}

		return images;
	}
	
	/**
	 * Read MNIST labels
	 * 
	 * @param filename File name
	 * @return Label array
	 * @throws IOException
	 */
	public static int[] readLabels (String filename)
		throws IOException
	{
		FileInputStream file = null;
		InputStream gzip = null;
		DataInputStream data = null;
		int labels[] = null;

		try {
			ForwardFeedNeuralNetwork mnistDatabase = new ForwardFeedNeuralNetwork();
			ClassLoader classLoader = mnistDatabase.getClass().getClassLoader();
			file = new FileInputStream(classLoader.getResource(filename).getFile());
			gzip = new GZIPInputStream(file);
			data = new DataInputStream(gzip);

			log.info("Reading MNIST labels...");

			int magicNumber = data.readInt();

			if (magicNumber!=2049) // 0x00000801 == 08 (unsigned byte) + 01 (vector)
				throw new IOException("Error while reading MNIST labels from "+filename);

			int size = data.readInt();
			
			labels = new int[size];

			for (int i=0; i<size; i++)
				labels[i] = data.readUnsignedByte();

			log.info("MNIST labels read from "+filename);

		} finally {
			
			if (data!=null)
				data.close();
			if (gzip!=null)
				gzip.close();
			if (file!=null)
				file.close();
		}
		
		return labels;
	}

	
	/**
	 * Normalize raw image data, i.e. convert to floating-point and rescale to [0,1].
	 * 
	 * @param image Raw image data
	 * @return Floating-point 2D array
	 */
	public static float[][] normalize (int image[][])
	{
		int rows = image.length;
		int columns = image[0].length;
		float data[][] = new float[rows][columns];
		
		for (int i=0; i<rows; i++)
			for (int j=0; j<rows; j++)
				data[i][j] = (float)image[i][j] / 255f;
		
		return data;
	}

	
	// Standard I/O
	
	public static String toString (int label)
	{
		return Integer.toString(label);
	}
	
	public static String toString (int image[][])
	{
		StringBuilder builder = new StringBuilder();

		for (int i=0; i<image.length; i++) {
			for (int j=0; j<image[i].length; j++) {
				String hex = Integer.toHexString(image[i][j]);
				if (hex.length()==1) 
					builder.append("0");
				builder.append(hex);
				builder.append(' ');				
			}
			builder.append('\n');
		}
		
		return builder.toString();
	}

	public static String toString (float image[][])
	{
		StringBuilder builder = new StringBuilder();

		for (int i=0; i<image.length; i++) {
			for (int j=0; j<image[i].length; j++) {
				builder.append( String.format(Locale.US, "%.3f ", image[i][j]) );
			}
			builder.append('\n');
		}
		
		return builder.toString();
	}

	// Test program
	
	public static void main (String[] args)
		throws IOException
	{
//		System.out.println("Working Directory = " +
//				System.getProperty("user.dir"));
////		 downloadMNIST("data/mnist/");
//
//
//		int images[][][];
//		//images = readImages("data/mnist/"+trainingImages);
//		images = readImages(testImages);
//
//		for (int i = 0; i <= 50; i++) {
//			System.out.println("Raw image data:");
//			System.out.println(toString(images[i]));
//
//			// Normalize image data
//			float data[][] = normalize(images[i]);
//
//			System.out.println("Normalized image:");
//			System.out.println(toString(data));
//
//			int labels[];
//			// labels = readLabels("data/mnist/"+trainingLabels);
//			labels = readLabels(testLabels);
//
//			System.out.println("Image label:");
//			System.out.println(labels[i]);
//		}


		final int numRows = 28; // The number of rows of a matrix.
		final int numColumns = 28; // The number of columns of a matrix.
		int outputNum = 10; // Number of possible outcomes (e.g. labels 0 through 9).
		int batchSize = 128; // How many examples to fetch with each step.
		int rngSeed = 123; // This random-number generator applies a seed to ensure that the same initial weights are used when training. We’ll explain why this matters later.
		int numEpochs = 15; // An epoch is a complete pass through a given dataset.

		//Get the DataSetIterators:
		DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(rngSeed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(1)
				.learningRate(0.006)
				.updater(Updater.NESTEROVS).momentum(0.9)
				.regularization(true).l2(1e-4)
				.list()
				.layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
						.nIn(numRows * numColumns)
						.nOut(50)
						.activation("relu")
						.weightInit(WeightInit.XAVIER)
						.build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
						.nIn(50)
						.nOut(outputNum)
						.activation("softmax")
						.weightInit(WeightInit.XAVIER)
						.build())
				.pretrain(false).backprop(true) //use backpropagation to adjust weights
						.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		Stopwatch timer = new Stopwatch();
		timer.start();
		model.init();
		//print the score with every 1 iteration
		model.setListeners(new ScoreIterationListener(1));

		log.info("Train model....");
		for( int i=0; i<numEpochs; i++ ){
			model.fit(mnistTrain);
		}


		log.info("Evaluate model....");
		Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
		while(mnistTest.hasNext()){
			DataSet next = mnistTest.next();
			INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
			eval.eval(next.getLabels(), output); //check the prediction against the true class
		}

		log.info(eval.stats());
		log.info("****************Example finished********************");
		timer.stop();
		long totalSeconds = timer.elapsedTime(TimeUnit.SECONDS);
		long excessSeconds = totalSeconds % 60;
		long minutes = (totalSeconds - excessSeconds) / 60;
		log.info("Total time: " + (minutes < 10 ? "0" + minutes : minutes) + ":" + (excessSeconds < 10 ? "0" + excessSeconds : excessSeconds));
	}

}
