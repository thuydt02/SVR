/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.

The class will load the 02 models in files and predict on the test dataset

args[0]: the X-model file name, default: model/X_model.txt
args[1]: the Y-model file name, default: model/Y_model.txt
args[2]: the test file name, default: data/forPrediction/10_feature_vectors.txt
args[3]: the predicted location file name, default: data/forPrediction/predicted_location.csv
model/X_model.txt model/Y_model.txt data/forPrediction/10_feature_vectors.txt data/forPrediction/predicted_location.csv
 */
package svm;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import static java.lang.System.exit;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.misc.SerializedClassifier;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

/**
 *
 * @author Do Thanh Thuy
 */
public class doPrediction {
    private static SerializedClassifier X_trained_model;
    private static SerializedClassifier Y_trained_model;
    private static Instances test_dataset;
    public static void main(String[] args) throws IOException, Exception{
        
        String X_model_fname = args[0];
        String Y_model_fname = args[1];
        String test_fname = args[2];
        String location_fname = args[3];
        X_trained_model = new SerializedClassifier();
        X_trained_model.setModelFile(new File(X_model_fname)); 
        Y_trained_model = new SerializedClassifier();
        Y_trained_model.setModelFile(new File(Y_model_fname)); 
        test_dataset = read_test_data(test_fname);
        output_predicted_location(location_fname);
    }            
    public static void output_predicted_location(String output_prediction_fname) throws Exception{        
        double[] X_ret_val = do_prediction(X_trained_model, test_dataset);
        double[] Y_ret_val = do_prediction(Y_trained_model, test_dataset);                       
        try (FileWriter predict_fout = new FileWriter(output_prediction_fname, false)) {        
            int n = Y_ret_val.length;
            for (int i = 0;i<n;i++){
                String s = String.valueOf(X_ret_val[i]) + "," + String.valueOf(Y_ret_val[i]) + "\n";
                predict_fout.write(s);
            }
            predict_fout.close();
        }catch (Exception e)        {
            System.out.println(e.getMessage());
            exit(0);
        }
    }                    
    private static Instances read_test_data(String fname_in) throws IOException {
        String tmp_csv_fname = fname_in + ".csv";
        create_input_csv_files(fname_in, tmp_csv_fname);
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(tmp_csv_fname));
        return loader.getDataSet();        
    }
    private static double[] do_prediction(SerializedClassifier the_model, Instances the_test) throws Exception{
        int n=the_test.numInstances();
        the_test.setClassIndex(the_test.numAttributes()-1);
        Instances labeled = new Instances(the_test);
        double[] ret_val = new double[n];
        for (int i = 0; i < n; i++)
        {        
            double[] predictionDistribution = the_model.distributionForInstance(the_test.instance(i)); 
            //System.out.printf("%5.20f %10.20f\n", test_set.instance(i).value(test_set.numAttributes()-1), predictionDistribution[0]); 
            ret_val[i] = predictionDistribution[0];
        }           
        return ret_val;
    }
    
    
    
    public static void create_input_csv_files(String test_file, String test_output_file) throws IOException{
                    
            Scanner test_scan; FileWriter test_fout;
            try {
                test_scan = new Scanner(new File(test_file));                
                test_fout = new FileWriter(test_output_file, false);

                if(test_scan.hasNextLine()){
                    if (test_scan.hasNextLine()){
                        String test_line = test_scan.nextLine();                        
                        String s = test_line + ",0\n";
                        String[] s_array = test_line.split(",");
                        int nn = s_array.length; String header = "";
                        for (int i = 1; i<=nn; i++) header = header + "F" + String.valueOf(i) + ",";
                        header = header + "Prediction\n";
                        test_fout.write(header);
                        test_fout.write(s);                        
                    }
                }            
                while (test_scan.hasNextLine()){
                    if (test_scan.hasNextLine()){
                        String test_line = test_scan.nextLine();                        
                        String s = test_line + ",0\n";                        
                        test_fout.write(s);                        
                    }
                }                
                test_fout.close();                
            } catch (FileNotFoundException ex) {
                Logger.getLogger(SVR.class.getName()).log(Level.SEVERE, null, ex);
            }        
    }
}
