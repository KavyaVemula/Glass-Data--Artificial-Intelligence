
package glassData.concrete;

import java.util.Date;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import process.plugin.AbstractIris;
import process.util.Helper;

/**
 *
 * @author kavyareddy
 */
public class glassData extends AbstractIris {
    
    //int numTestRows = -1;
    int numCols = Helper.headers.size()-1;
    
    public glassData() {
        super("glassData", "glassData.csv");
        int numTestRows = super.numTestRows;
    }

    @Override
    public void createNetwork() {
        network = new BasicNetwork();
        // Creates an input layer with 10 nodes
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 10));
        // Creates a hidden layer with 20 nodes
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 20));
        // Creates output layer with 2 nodes
        network.addLayer(new BasicLayer(new ActivationTANH(), false, 2));
        network.getStructure().finalizeStructure();
        network.reset();        
    }

    @Override
    public void testNetwork() {
 
        BasicMLDataSet testSet;
                
        int numCols = Helper.headers.size() - 1;
        double[][] testInputs = new double[numTestRows][numCols];
        double[][] testIdeals = new double[numTestRows][numCols];
        
        for(int row=testStart; row <= testEnd; row++){
            for(int col=0; col < numCols; col++){
                testInputs[row - testStart][col] = allInputs[row][col];       
            }
        }   

        numCols = equilateral.encode(0).length;  
         
         for(int row=testStart; row <= testEnd; row++) {
            for(int col=0; col < numCols; col++){
                testIdeals[row - testStart][col] = allIdeals[row][col];      
            }
        }
         
        testSet = new BasicMLDataSet(testInputs, testIdeals);
         
        int count = 0;
        
        Date date = new Date();
        
        System.out.print("Glass data ANN Report " + date.toString() +"\n");
        
        String header = String.format("%-5s %-5s %-6s", "Actual","Predicted","Test");
        System.out.println(header);
        for(MLDataPair pair : testSet){
            
           MLData predicted = network.compute(pair.getInput());
           int predictedNumber = equilateral.decode(predicted.getData());
           subtypes.get(predictedNumber);
           
           MLData ideal = network.compute(pair.getIdeal());
           int idealNumber = equilateral.decode(ideal.getData());
           subtypes.get(idealNumber);
           
           if(subtypes.get(predictedNumber).equals(subtypes.get(idealNumber))){
               count++;
                System.out.println(subtypes.get(predictedNumber)+" "+subtypes.get(idealNumber)+" Passed");
           }
           else
                System.out.println(subtypes.get(predictedNumber)+" "+subtypes.get(idealNumber)+" Fail");
        } 
        System.out.println((double)count / (double)(testEnd - testStart) * 100);
    }
}
