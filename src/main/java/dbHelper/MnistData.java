package dbHelper;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

/**
 * Created by Mike on 04/12/2016.
 */
public class MnistData {

    private DataSetIterator mnistTrain;
    private DataSetIterator mnistTest;

    public MnistData(int batchSize, int rngSeed) {
        try {
            setMnistTrain(new MnistDataSetIterator(batchSize, true, rngSeed));
            setMnistTest(new MnistDataSetIterator(batchSize, false, rngSeed));
        } catch (IOException e) {

        }
    }

    public DataSetIterator getMnistTrain() {
        return mnistTrain;
    }

    public void setMnistTrain(DataSetIterator mnistTrain) {
        this.mnistTrain = mnistTrain;
    }

    public DataSetIterator getMnistTest() {
        return mnistTest;
    }

    public void setMnistTest(DataSetIterator mnistTest) {
        this.mnistTest = mnistTest;
    }
}
