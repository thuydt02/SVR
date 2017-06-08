/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.

The class is to train the train dataset and get 02 models: X_model and Y_model
The models can be chosen with the smallest RRSE on cross-validation or test prediction

args[0]: The whole train dataset of feature vectors (no class attribute), default: data/D2N100_euclidean_exp_W20_m2_MDS.X
args[1]: The class attributes corresponding to the feature vectors(X_class attribute and Y_class attribute), default: data/D2N100
args[2]: The test dataset: line number of feature vectors should be picked up for the test set (normally 20%), default: data/L20N100
args[3]: The train dataset constructed from args[0], args[1], args[2], default: data/feature_label_points.csv
args[4]: The test dataset cinstructed from args[0], args[1], args[2], default: data/test_points.csv
args[5]: The X model file, default: model/X_model.txt
args[6]: The Y model file, default: model/Y_model.txt
args[7]: The outcomes of prediction on the test set, default: predictResults/predicted_location.txt
data/D2N100_euclidean_exp_W20_m2_MDS.X data/D2N100 data/L20N100 data/feature_label_points.csv data/test_points.csv model/X_model.txt model/Y_model.txt predictResults/predicted_location.txt
*/
package svm;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import static java.lang.System.exit;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.core.Debug;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.functions.LibSVM;       
import weka.core.Utils;
import weka.core.Debug.Random;
import weka.core.converters.CSVLoader;


/**
 *
 * @author Thuy Do
 */
public class SVR {
    private static Instances train_dataset;
    private static Instances test_dataset;

    public static void main(String[] args) throws IOException, Exception {
        String feature_file = args[0];
        String lable_file = args[1];
        String test_file = args[2];
        String train_output_file = args[3];
        String test_output_file = args[4];
        String X_model_fname = args[5];
        String Y_model_fname = args[6];
        String predict_fname = args[7];
        create_input_csv_files(feature_file, lable_file, test_file, train_output_file, test_output_file);
        read_csv_data_to_train_test_datasets(train_output_file, test_output_file);                
        String[] best_c_g_in_loose_grid = new String[6];        
        best_c_g_in_loose_grid = find_best_c_gamma_for_XY_in_loose_grid(train_dataset);
        System.out.println("Best option for X: " + best_c_g_in_loose_grid[0]);
        System.out.println("Best option for Y: " + best_c_g_in_loose_grid[3]);        
//best_c_g_in_loose_grid[0] = "-C 2048.0 -G 0.5 -S 4 -K 2 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";//X                
//best_c_g_in_loose_grid[3] = "-C 8192.0 -G 0.5 -S 4 -K 2 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";//Y                       
        String[] best_nu = find_best_nu(train_dataset, best_c_g_in_loose_grid[0], best_c_g_in_loose_grid[3]);
        System.out.println("Best NU option for X: " + best_nu[0]);
        System.out.println("Best NU option for Y: " + best_nu[1]);
        double[] ic = new double[2]; double[] igamma=new double[2];
        ic[0] = Double.parseDouble(best_c_g_in_loose_grid[1]);ic[1] = Double.parseDouble(best_c_g_in_loose_grid[4]);
        igamma[0] = Double.parseDouble(best_c_g_in_loose_grid[2]);igamma[1] = Double.parseDouble(best_c_g_in_loose_grid[5]);       
        String[] best_c_g_in_finer_grid = find_best_c_gamma_for_XY_finner_grid(train_dataset,best_nu,ic,igamma);
        //String[] best_c_g_in_finer_grid = find_best_c_gamma_for_XY_finner_grid_test_instance(best_nu,ic,igamma);
        System.out.println("Best finer grid option for X: " + best_c_g_in_finer_grid[0]);
        System.out.println("Best finer grid option for Y: " + best_c_g_in_finer_grid[1]);                
String X_options;// = "-C 7643.406266669453 -G 0.3298769776932151 -N 0.345 -S 4 -K 2 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";
String Y_options;// = "-C 8191.99999999996 -G 0.4999999999999874 -N 0.373 -S 4 -K 2 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";

        X_options = best_c_g_in_finer_grid[0]; Y_options = best_c_g_in_finer_grid[1];
        evaluate_models_test_instances(X_model_fname, Y_model_fname, predict_fname,X_options,Y_options);
//evaluate_models(X_model_fname, Y_model_fname, predict_fname,X_options,Y_options);
        

        
    }
       
    
    
    public static void evaluate_models(String X_model_fname, String Y_model_fname, String output_prediction_fname, String X_options, String Y_options) throws Exception{
        //String options = "-C 8192.0 -G 0.5 -S 4 -K 2 -N 0.4 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";
        
        String options = "-C 7643.406266669453 -G 0.3298769776932151 -N 0.345 -S 4 -K 2 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";
        Instances X_train_dataset = new Instances(remove_feature(train_dataset, train_dataset.numAttributes()));
        Instances X_test_dataset = new Instances(remove_feature(test_dataset, test_dataset.numAttributes()));
        double[] X_ret_val = do_classifying_predicting(X_train_dataset, X_test_dataset, X_train_dataset.numAttributes()-1, X_options,X_model_fname);        
        //options = "-C 8192.0 -G 0.5 -S 4 -K 2 -N 0.4 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";
        options = "-C 8191.99999999996 -G 0.4999999999999874 -N 0.373 -S 4 -K 2 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";
        Instances Y_train_dataset = new Instances(remove_feature(train_dataset, train_dataset.numAttributes()-1));
        Instances Y_test_dataset = new Instances(remove_feature(test_dataset, test_dataset.numAttributes()-1));
        double[] Y_ret_val = do_classifying_predicting(Y_train_dataset, Y_test_dataset, Y_train_dataset.numAttributes()-1, Y_options,Y_model_fname);
        int ind_true_X = test_dataset.numAttributes()-2, ind_true_Y = test_dataset.numAttributes()-1;
        try (FileWriter predict_fout = new FileWriter(output_prediction_fname, false)) {
            double tmp_MAE = 0, tmp_RMSE = 0, tmp_RAE_numerator = 0, tmp_RAE_denominator=0;
            double tmp_RRSE_numerator = 0, tmp_RRSE_denominator=0,tmp_mean_X = 0, tmp_mean_Y = 0;
            int n = Y_ret_val.length;
            for (int i = 0;i<n;i++){
                double true_X = test_dataset.instance(i).value(ind_true_X);
                double true_Y = test_dataset.instance(i).value(ind_true_Y);
                tmp_MAE = tmp_MAE + Math.abs(true_X - X_ret_val[i]) + Math.abs(true_Y - Y_ret_val[i]);
                tmp_RMSE = tmp_RMSE + Math.pow((true_X - X_ret_val[i]), 2) + Math.pow((true_Y - Y_ret_val[i]), 2);
                tmp_mean_X += true_X;tmp_mean_Y += true_Y;
                String s = String.valueOf(X_ret_val[i]) + "," + String.valueOf(Y_ret_val[i]) + "\n";
                predict_fout.write(s);
            }   
            double mean_X = 0, mean_Y=0;            
            for (int i=0;i<n;i++){
                double true_X = test_dataset.instance(i).value(ind_true_X);
                double true_Y = test_dataset.instance(i).value(ind_true_Y);
                tmp_RAE_numerator = tmp_RAE_numerator + Math.abs(true_X - X_ret_val[i]) + Math.abs(true_Y - Y_ret_val[i]);
                tmp_RAE_denominator = tmp_RAE_denominator + Math.abs(mean_X - true_X) + Math.abs(mean_Y-true_Y);                
                tmp_RRSE_numerator = tmp_RRSE_numerator + Math.pow((true_X - X_ret_val[i]),2) + Math.pow((true_Y - Y_ret_val[i]),2);
                tmp_RRSE_denominator = tmp_RRSE_denominator + Math.pow((mean_X - true_X),2) + Math.pow((mean_Y-true_Y),2);
            }   double MAE = 0,RMSE = 0;
            if (n>0){ MAE= tmp_MAE/n; RMSE = Math.pow(tmp_RMSE/n, 0.5);mean_X=tmp_mean_X/n; mean_Y=tmp_mean_Y/n;}
            double RAE=0, RRSE=0;
            if (tmp_RAE_denominator >0){RAE = tmp_RAE_numerator/tmp_RAE_denominator;}
            if (tmp_RRSE_denominator >0){RRSE = Math.pow(tmp_RRSE_numerator/tmp_RRSE_denominator,0.5);}
            String s="\n";
            predict_fout.write(s);
            s="Mean absolute error: "+String.valueOf(MAE)+"\n";
            predict_fout.write(s);
            s="Root mean squared error: "+String.valueOf(RMSE)+"\n";
            predict_fout.write(s);
            s="Relative absolute error: "+String.valueOf(RAE*100)+" %\n";
            predict_fout.write(s);
            s="Root relative squared error: "+String.valueOf(RRSE*100)+" %\n";
            predict_fout.write(s);
        }catch (Exception e)        {
            System.out.println(e.getMessage());
            exit(0);
        }
    }
    
    public static double building_c_svc_model(Instances dataset, int class_index, String options){
        System.out.println("----------------------------------------------------------------------------------------------------------");               
        FilteredClassifier fc = new FilteredClassifier();
        dataset.setClassIndex(class_index);     
        LibSVM svm = new LibSVM();        
        try{
            
            svm.setOptions(Utils.splitOptions(options));
            System.out.println(options);
               //fc.setClassifier(svm);            
            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(svm, dataset, 3, new Random(1));
            //fc.buildClassifier(dataset);
            svm.buildClassifier(dataset);
            //System.out.println("Percent correct: " + Double.toString(eval.pctCorrect()));
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));  
            //System.out.println("correct: " + String.valueOf(eval.correct()));
            //System.out.println("correlationCoefficent:" + String.valueOf(eval.correlationCoefficient()));
        //    return eval.correlationCoefficient();
            return eval.rootRelativeSquaredError();
            
        } catch(Exception ee){
            System.out.println(ee.getMessage());
            exit(0);
        }
        return -1;
    }
    
    public static String[] find_best_c_gamma_for_XY_in_loose_grid(Instances train_dataset) throws Exception{        
        //train_dataset = remove_feature(train_dataset, train_dataset.numAttributes()-1); //remove Y        
        int ic = -15; int max_ic = 25;
        int igamma = -15; int max_igamma = 15;
        double e = 10e-5; double r = 10e-5; int k = 2; double nu = 0.4, min_nu = 0.00001; int S=4;
        System.out.println("-------------------------------------------------------------------------------");
        System.out.println("Regressing with class attibute X .................");
        System.out.println("-------------------------------------------------------------------------------");
        Instances tmp = new Instances(remove_feature(train_dataset, train_dataset.numAttributes())); //remove Y                                   
        int class_index = tmp.numAttributes()-1;
        String best_option_X = ""; double max_correlationeffiecient_X = -2;
        double min_RRSE_X = 1000000; int ic_X_ret = ic; int igamma_X_ret = igamma;
        while (ic <= max_ic) {
            igamma = -15;
            while (igamma < max_igamma){
                double c = Math.pow(2, ic); double g = Math.pow(2, igamma);
                String c_str = String.valueOf(c);
                String gamma_str = String.valueOf(g);
                String c_gammma_option = "-C " + c_str + " -G " + gamma_str;
                String options = c_gammma_option + " -S " + String.valueOf(S) + " -K " + String.valueOf(k) + " -N " + String.valueOf(nu)+ " -E " + String.valueOf(e) + " -R " + String.valueOf(r) + " -W 1 -Z";
                double ret_val = building_c_svc_model(tmp, class_index, options);
                //if (max_correlationeffiecient_X < ret_val) {max_correlationeffiecient_X = ret_val;best_option_X = options;}                
                if (min_RRSE_X > ret_val) {min_RRSE_X = ret_val;best_option_X = options; ic_X_ret = ic; igamma_X_ret=igamma;}
                igamma = igamma + 2;
            }
            ic = ic + 2;
        }
        
        System.out.println("-------------------------------------------------------------------------------");
        System.out.println("Regressing with class attibute Y .................");
        System.out.println("-------------------------------------------------------------------------------");
        ic = -15; max_ic = 25; int ic_Y_ret = ic;int igamma_Y_ret = igamma;
        max_igamma = 15;
        e = 10e-5; r = 10e-5; k = 2; nu = 0.4; min_nu = 0.00001; S=4;
        tmp = new Instances(remove_feature(train_dataset, train_dataset.numAttributes()-1)); //remove X                                   
        class_index = tmp.numAttributes()-1;
        String best_option_Y = ""; double max_correlationeffiecient_Y = -2;
        double min_RRSE_Y = 1000000;
        while (ic <= max_ic) {
            igamma = -15;
            while (igamma < max_igamma){
                double c = Math.pow(2, ic); double g = Math.pow(2, igamma);
                String c_str = String.valueOf(c);
                String gamma_str = String.valueOf(g);
                String c_gammma_option = "-C " + c_str + " -G " + gamma_str;
                String options = c_gammma_option + " -S " + String.valueOf(S) + " -K " + String.valueOf(k) + " -N " + String.valueOf(nu)+ " -E " + String.valueOf(e) + " -R " + String.valueOf(r) + " -W 1 -Z";
                double ret_val = building_c_svc_model(tmp, class_index, options);
                //if (max_correlationeffiecient_Y < ret_val) {max_correlationeffiecient_Y = ret_val;best_option_Y = options;}                
                if (min_RRSE_Y > ret_val) {min_RRSE_Y = ret_val;best_option_Y = options;ic_Y_ret=ic;igamma_Y_ret=igamma;}                
                igamma = igamma + 2;
            }
            ic = ic + 2;
        }
        //System.out.println("best_option_X: " + best_option_X + " CorrelationEffiecent: " + String.valueOf(max_correlationeffiecient_X));
        //System.out.println("best_option_Y: " + best_option_Y + " CorrelationEffiecent: " + String.valueOf(max_correlationeffiecient_Y));
        String[] ret_val = new String[6];
        ret_val[0] = best_option_X; ret_val[1] = String.valueOf(ic_X_ret); ret_val[2] = String.valueOf(igamma_X_ret);
        ret_val[3]=best_option_Y; ret_val[4] = String.valueOf(ic_Y_ret); ret_val[5] = String.valueOf(igamma_Y_ret);return ret_val;
    }
    
    public static String[] find_best_c_gamma_for_XY_finner_grid(Instances train_dataset, String[] options, double[] ic, double[] igamma) throws Exception{
        String[] ret_val = new String[2];
        Instances X_train_dataset = new Instances(remove_feature(train_dataset, train_dataset.numAttributes()));        
        //String options = "-N 0.345 -S 4 -K 2 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";
        //ret_val[0] = find_best_c_gamma_finer_grid(X_train_dataset, X_train_dataset.numAttributes()-1, 11, -1, options[0]); //-C 2048 -G 0.5
        ret_val[0] = find_best_c_gamma_finer_grid(X_train_dataset, X_train_dataset.numAttributes()-1, ic[0], igamma[0], options[0]); //-C 2048 -G 0.5
        //options = "-N 0.373 -S 4 -K 2 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";
        Instances Y_train_dataset = new Instances(remove_feature(train_dataset, train_dataset.numAttributes()-1));        
        //ret_val[1] = find_best_c_gamma_finer_grid(Y_train_dataset, Y_train_dataset.numAttributes()-1,13, -1,options[1] ); //-C 8192 -G 0.5
        ret_val[1] = find_best_c_gamma_finer_grid(Y_train_dataset, Y_train_dataset.numAttributes()-1,ic[1],igamma[1],options[1] ); //-C 8192 -G 0.5
        return ret_val;

    }
    
    
    public static String find_best_c_gamma_finer_grid(Instances dataset, int class_index, double ind_c, double ind_g, String options) throws Exception{        
            
        double ic = ind_c - 2; double max_ic = ind_c + 2;
        double igamma = ind_g -2; double max_igamma = ind_g + 2;
        System.out.println("-------------------------------------------------------------------------------");
        System.out.println("Regressing..................................");
        System.out.println("-------------------------------------------------------------------------------");
//remove -c and -g in options if it has           
        String st1="";       
        int c_ind = options.indexOf("-C");
        if (c_ind != -1){
            int start = c_ind + 3; int j = start;
            while (j < options.length()){
                if (options.charAt(j) != ' ') j++;
                else break;
            }
            if (c_ind == 0) st1 = options.substring(j+1);
            else            st1 = options.substring(0, c_ind-1)  + options.substring(j);            
        }        
        options = st1;
        System.out.println(options);
        
        int g_ind = options.indexOf("-G");
        if (g_ind != -1){
            int start = g_ind + 3; int j = start;
            while (j < options.length()){
                if (options.charAt(j) != ' ') j++;
                else break;
            }
            if (g_ind == 0) st1 = options.substring(j+1);
            else st1 = options.substring(0, g_ind-1)  + options.substring(j);            
        }
        options = st1;
//end of remove -c -g in options                
        String best_options = ""; double max_correlationeffiecient_X = -2;
        double min_RRSE = 1000000;
        while (ic <= max_ic) {
            igamma = -15;
            while (igamma < max_igamma){
                double c = Math.pow(2, ic); double g = Math.pow(2, igamma);             
                String new_options = "-C " + String.valueOf(c) + " -G " + String.valueOf(g) + " "+ options;
                double ret_val = building_c_svc_model(dataset, class_index, new_options);
                if (min_RRSE > ret_val) {min_RRSE = ret_val;best_options = new_options;}                
                igamma = igamma + 0.1;
            }
            ic = ic + 0.1;
        }        
        //System.out.println("best_option: " + best_options );
        return best_options;    
    }
    public static String find_best_nu_for_svc_svr(Instances train_dataset, int class_index, String options) throws Exception{        
        //int class_index = 2;        
        double nu = 0.4; double nu_min = 0.1e-5;
        double min_RRSE = 1000000;
        String best_options = "";
        String[] option_arr = options.split("-");
        options = "";    
        for (int i = 0; i<option_arr.length; i++){
            if (-1 == option_arr[i].indexOf("N") && (!option_arr[i].isEmpty()) ) {                        
            options = options + "-"+ option_arr[i];}
        }                
        while (nu > nu_min) {            
            String new_options = "-N " + String.valueOf(nu) + " " + options;
            double RRSE = building_c_svc_model(train_dataset, class_index, new_options);
            if (min_RRSE > RRSE) {min_RRSE = RRSE; best_options = new_options;}
            nu = nu - 0.001;
        }
        return best_options;
    }
    
    
    private static double[] do_classifying_predicting(Instances dataset, Instances test_set, int class_index, String options, String fout) throws Exception
    {   System.out.println("------------------------------------------------------------------");        
        System.out.println("Building the model and predicting is in progress ...");
        System.out.println("------------------------------------------------------------------");        
        //FilteredClassifier fc = new FilteredClassifier();
        dataset.setClassIndex(class_index);
        
        LibSVM svm = new LibSVM();
        
        try{            
            svm.setOptions(Utils.splitOptions(options));

            //fc.setClassifier(svm);
            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(svm, dataset, 3, new Random(1));
            //fc.buildClassifier(dataset);
            svm.buildClassifier(dataset);
            String model_fname = fout; 
            save_LibSVM_to_file(svm, fout);
            //--------------------
            out_put_outcomes(eval);
        //    System.out.println("\n\n" +svm.toString());
        } catch(Exception e){
            System.out.println(e.getMessage());
            exit(0);
        }
        test_set.setClassIndex(class_index);
        int k=0;int n=test_set.numInstances();
        Instances tmp = new Instances(test_set);        
        System.out.println("---------------------------------------------------");
        int numTestInstances = test_set.numInstances();
        System.out.printf("There are %d test instances\n", numTestInstances);
        
    // Loop over each test instance.
        System.out.println("Expected                         Predicted");
        int count = 0;
        double ret_val[] = new double[numTestInstances];
        for (int i = 0; i < numTestInstances; i++)
        {        
        // Make the prediction here.           
        //   double predictionIndex = svm.classifyInstance(test_set.instance(i));         
        // Get the prediction probability distribution.
            double[] predictionDistribution = svm.distributionForInstance(test_set.instance(i)); 
            System.out.printf("%5.20f %10.20f\n", test_set.instance(i).value(test_set.numAttributes()-1), predictionDistribution[0]); 
            ret_val[i] = predictionDistribution[0];
        } 
        
        return ret_val;
    }    
    
    public static void evaluate_models_test_instances(String X_model_fname, String Y_model_fname, String output_prediction_fname, String X_options, String Y_options) throws Exception{
      
        Instances X_train_dataset = new Instances(remove_feature(train_dataset, train_dataset.numAttributes()));
        Instances X_test_dataset = new Instances(remove_feature(test_dataset, test_dataset.numAttributes()));
        double[] X_ret_val = do_predicting(X_train_dataset, X_test_dataset, X_train_dataset.numAttributes()-1, X_options,X_model_fname);                        
      
        Instances Y_train_dataset = new Instances(remove_feature(train_dataset, train_dataset.numAttributes()-1));
        Instances Y_test_dataset = new Instances(remove_feature(test_dataset, test_dataset.numAttributes()-1));
        double[] Y_ret_val = do_predicting(Y_train_dataset, Y_test_dataset, Y_train_dataset.numAttributes()-1, Y_options,Y_model_fname);
        
        int ind_true_X = test_dataset.numAttributes()-2, ind_true_Y = test_dataset.numAttributes()-1;
        try (FileWriter predict_fout = new FileWriter(output_prediction_fname, false)) {
            double tmp_MAE = 0, tmp_RMSE = 0, tmp_RAE_numerator = 0, tmp_RAE_denominator=0;
            double tmp_RRSE_numerator = 0, tmp_RRSE_denominator=0,tmp_mean_X = 0, tmp_mean_Y = 0;
            int n = Y_ret_val.length-4;
            for (int i = 0;i<n;i++){
                double true_X = test_dataset.instance(i).value(ind_true_X);
                double true_Y = test_dataset.instance(i).value(ind_true_Y);
                tmp_MAE = tmp_MAE + Math.abs(true_X - X_ret_val[i]) + Math.abs(true_Y - Y_ret_val[i]);
                tmp_RMSE = tmp_RMSE + Math.pow((true_X - X_ret_val[i]), 2) + Math.pow((true_Y - Y_ret_val[i]), 2);
                tmp_mean_X += true_X;tmp_mean_Y += true_Y;
                String s = String.valueOf(X_ret_val[i]) + "," + String.valueOf(Y_ret_val[i]) + "\n";
                predict_fout.write(s);
            }   
            double mean_X = 0, mean_Y=0;            
            for (int i=0;i<n;i++){
                double true_X = test_dataset.instance(i).value(ind_true_X);
                double true_Y = test_dataset.instance(i).value(ind_true_Y);
                tmp_RAE_numerator = tmp_RAE_numerator + Math.abs(true_X - X_ret_val[i]) + Math.abs(true_Y - Y_ret_val[i]);
                tmp_RAE_denominator = tmp_RAE_denominator + Math.abs(mean_X - true_X) + Math.abs(mean_Y-true_Y);                
                tmp_RRSE_numerator = tmp_RRSE_numerator + Math.pow((true_X - X_ret_val[i]),2) + Math.pow((true_Y - Y_ret_val[i]),2);
                tmp_RRSE_denominator = tmp_RRSE_denominator + Math.pow(mean_X - true_X,2) + Math.pow(mean_Y-true_Y,2);
            }   double MAE = 0,RMSE = 0;
            if (n>0){ MAE= tmp_MAE/n; RMSE = Math.pow(tmp_RMSE/n, 0.5);mean_X=tmp_mean_X/n; mean_Y=tmp_mean_Y/n;}
            double RAE=0, RRSE=0;
            if (tmp_RAE_denominator >0){RAE = tmp_RAE_numerator/tmp_RAE_denominator;}
            if (tmp_RRSE_denominator >0){RRSE = Math.pow(tmp_RRSE_numerator/tmp_RRSE_denominator,0.5);}
            String s="\nSummary ON X\n";
            predict_fout.write(s);
            s="Mean absolute error: "+String.valueOf(X_ret_val[n])+"\n"; predict_fout.write(s);
            s="Root mean squared error: "+String.valueOf(X_ret_val[n+1])+"\n"; predict_fout.write(s);
            s="Relative absolute error: "+String.valueOf(X_ret_val[n+2]*100)+" %\n"; predict_fout.write(s);
            s="Root relative squared error: "+String.valueOf(X_ret_val[n+3]*100)+" %\n"; predict_fout.write(s);
            
            s="Summary ON Y\n";
            predict_fout.write(s);
            s="Mean absolute error: "+String.valueOf(Y_ret_val[n])+"\n"; predict_fout.write(s);
            s="Root mean squared error: "+String.valueOf(Y_ret_val[n+1])+"\n"; predict_fout.write(s);
            s="Relative absolute error: "+String.valueOf(Y_ret_val[n+2]*100)+" %\n"; predict_fout.write(s);
            s="Root relative squared error: "+String.valueOf(Y_ret_val[n+3]*100)+" %\n"; predict_fout.write(s);
            
            s="Summary ON XY\n";
            predict_fout.write(s);
            s="Mean absolute error: "+String.valueOf(MAE)+"\n";
            predict_fout.write(s);
            s="Root mean squared error: "+String.valueOf(RMSE)+"\n";
            predict_fout.write(s);
            s="Relative absolute error: "+String.valueOf(RAE*100)+" %\n";
            predict_fout.write(s);
            s="Root relative squared error: "+String.valueOf(RRSE*100)+" %";
            predict_fout.write(s);
        }catch (Exception e)        {
            System.out.println(e.getMessage());
            exit(0);
        }
    }
    
    
    
    public static String[] find_best_c_gamma_for_XY_finner_grid_test_instance(String[] options, double[] ic, double[] igamma) throws Exception{
        String[] ret_val = new String[2];
        Instances X_train_dataset = new Instances(remove_feature(train_dataset, train_dataset.numAttributes()));
        Instances X_test_set = new Instances(remove_feature(test_dataset, test_dataset.numAttributes()));        
        ret_val[0] = find_best_c_gamma_finer_grid_test_instances(X_train_dataset, X_test_set, X_test_set.numAttributes()-1, ic[0], igamma[0], options[0]);
        
        Instances Y_train_dataset = new Instances(remove_feature(train_dataset, train_dataset.numAttributes()-1));        
        Instances Y_test_set = new Instances(remove_feature(test_dataset, test_dataset.numAttributes()-1));        
        ret_val[1] = find_best_c_gamma_finer_grid_test_instances(Y_train_dataset,Y_test_set, Y_test_set.numAttributes()-1,ic[1],igamma[1],options[1] );
        return ret_val;

    }
    
    
    public static String find_best_c_gamma_finer_grid_test_instances(Instances dataset, Instances test_set, int class_index, double ind_c, double ind_g, String options) throws Exception{                    
        double ic = ind_c - 2; double max_ic = ind_c + 2;
        double igamma = ind_g -2; double max_igamma = ind_g + 2;
        System.out.println("-------------------------------------------------------------------------------");
        System.out.println("Regressing..................................");
        System.out.println("-------------------------------------------------------------------------------");
//remove -c and -g in options if it has           
        String st1="";       
        int c_ind = options.indexOf("-C");
        if (c_ind != -1){
            int start = c_ind + 3; int j = start;
            while (j < options.length()){
                if (options.charAt(j) != ' ') j++;
                else break;
            }
            if (c_ind == 0) st1 = options.substring(j+1);
            else            st1 = options.substring(0, c_ind-1)  + options.substring(j);            
            options = st1;
        }        
        
        System.out.println(options);
        
        int g_ind = options.indexOf("-G");
        if (g_ind != -1){
            int start = g_ind + 3; int j = start;
            while (j < options.length()){
                if (options.charAt(j) != ' ') j++;
                else break;
            }
            if (g_ind == 0) st1 = options.substring(j+1);
            else st1 = options.substring(0, g_ind-1)  + options.substring(j);            
            options = st1;
        }
        
//end of remove -c -g in options                
        String best_options = ""; double max_correlationeffiecient_X = -2;
        double min_RRSE = 1000000;
        while (ic <= max_ic) {
            igamma = -15;
            while (igamma < max_igamma){
                double c = Math.pow(2, ic); double g = Math.pow(2, igamma);             
                String new_options = "-C " + String.valueOf(c) + " -G " + String.valueOf(g) + " "+ options;
                double ret_val = do_predicting_RRSE(dataset, test_set, class_index, options);
                //double ret_val = building_c_svc_model(dataset, class_index, new_options);
                if (min_RRSE > ret_val) {min_RRSE = ret_val;best_options = new_options;}                
                igamma = igamma + 0.1;
            }
            ic = ic + 0.1;
        }                
        return best_options;    
    }
    
    
    
    private static double[] do_predicting(Instances dataset, Instances test_set, int class_index, String options, String model_fname) throws Exception
    {   //System.out.println("------------------------------------------------------------------");        
        //System.out.println("Building the model and predicting is in progress ...");
        //System.out.println("------------------------------------------------------------------");                
        dataset.setClassIndex(class_index);        
        LibSVM svm = new LibSVM();        
        try{            
            svm.setOptions(Utils.splitOptions(options));
            svm.buildClassifier(dataset);
            save_LibSVM_to_file(svm, model_fname);
        } catch(Exception e){
            System.out.println(e.getMessage());
            exit(0);
        }
        test_set.setClassIndex(class_index); int n=test_set.numInstances();
        //Instances tmp = new Instances(test_set);        
        System.out.println("---------------------------------------------------");
        
        //System.out.printf("There are %d test instances\n", n);
        int count = 0;
        double ret_val[] = new double[n + 4];
        double tmp_MAE = 0, tmp_RMSE = 0, tmp_RAE_numerator = 0, tmp_RAE_denominator=0;
        double tmp_RRSE_numerator = 0, tmp_RRSE_denominator=0,tmp_mean = 0;
        int ind_true_value = test_set.numAttributes()-1;
        for (int i = 0; i < n; i++)
        {        
            double[] predictionDistribution = svm.distributionForInstance(test_set.instance(i)); 
            ret_val[i] = predictionDistribution[0];
            double true_value = test_set.instance(i).value(ind_true_value);
            tmp_MAE = tmp_MAE + Math.abs(true_value - ret_val[i]);
            tmp_RMSE = tmp_RMSE + Math.pow((true_value - ret_val[i]), 2);
            tmp_mean += true_value;
        }
        double mean = 0, MAE = 0, RMSE = 0, RAE=0, RRSE=0;        
        if (n>0){ MAE= tmp_MAE/n; RMSE = Math.pow(tmp_RMSE/n, 0.5); mean=tmp_mean/n;}        
        for (int i=0;i<n;i++){
                double true_value = test_set.instance(i).value(ind_true_value);                
                tmp_RAE_numerator = tmp_RAE_numerator + Math.abs(true_value - ret_val[i]);
                tmp_RAE_denominator = tmp_RAE_denominator + Math.abs(mean - true_value);                
                tmp_RRSE_numerator = tmp_RRSE_numerator + Math.pow((true_value - ret_val[i]),2);
                tmp_RRSE_denominator = tmp_RRSE_denominator + Math.pow((mean - true_value),2);
            }        
        if (tmp_RAE_denominator >0){RAE = tmp_RAE_numerator/tmp_RAE_denominator;}
        if (tmp_RRSE_denominator >0){RRSE = Math.pow(tmp_RRSE_numerator/tmp_RRSE_denominator,0.5);}
        ret_val[n] = MAE;ret_val[n+1] = RMSE;ret_val[n+2]=RAE;ret_val[n+3]=RRSE;
        return ret_val;
    }    
    
    
    private static double do_predicting_RRSE(Instances dataset, Instances test_set, int class_index, String options) throws Exception
    {   
        dataset.setClassIndex(class_index);        
        LibSVM svm = new LibSVM();        
        try{            
            svm.setOptions(Utils.splitOptions(options));
            svm.buildClassifier(dataset);
        } catch(Exception e){
            System.out.println(e.getMessage());
            exit(0);
        }
        test_set.setClassIndex(class_index); int n=test_set.numInstances();        
        //System.out.println("---------------------------------------------------");                
        int count = 0;
        double ret_val[] = new double[n];
        //double tmp_MAE = 0, tmp_RMSE = 0, tmp_RAE_numerator = 0, tmp_RAE_denominator=0;
        double tmp_RRSE_numerator = 0, tmp_RRSE_denominator=0,tmp_mean = 0;
        int ind_true_value = test_set.numAttributes()-1;
        for (int i = 0; i < n; i++)
        {        
            double[] predictionDistribution = svm.distributionForInstance(test_set.instance(i)); 
            ret_val[i] = predictionDistribution[0];
            double true_value = test_set.instance(i).value(ind_true_value);
            //tmp_MAE = tmp_MAE + Math.abs(true_value - ret_val[i]);
            //tmp_RMSE = tmp_RMSE + Math.pow((true_value - ret_val[i]), 2);
            tmp_mean += true_value;
        }
        double mean = 0, RRSE=0;
        if (n>0){mean=tmp_mean/n;}        
        for (int i=0;i<n;i++){
                double true_value = test_set.instance(i).value(ind_true_value);                                                
                tmp_RRSE_numerator = tmp_RRSE_numerator + Math.pow((true_value - ret_val[i]),2);
                tmp_RRSE_denominator = tmp_RRSE_denominator + Math.pow((mean - true_value),2);
            }                
        if (tmp_RRSE_denominator >0){RRSE = Math.pow(tmp_RRSE_numerator/tmp_RRSE_denominator,0.5);}        
        return RRSE;
    }    
    
    
    public static void save_model_to_file(FilteredClassifier fc, String fname){
        Debug.saveToFile(fname, fc);                
    }
    public static void save_LibSVM_to_file(LibSVM lib, String fname){
        Debug.saveToFile(fname, lib);                
    }
    public static void out_put_outcomes(Evaluation eval) throws Exception{
        System.out.println("------------------------------------------------------------------");
        //System.out.println("Percent correct: " + Double.toString(eval.pctCorrect()));
        //System.out.println(eval.toMatrixString("Confusion Matrix"));
        System.out.println(eval.toSummaryString("Summary for the model", true));    
        
    }
    private static Instances delete_with_missing_value(Instances dataset){
        int n = dataset.numAttributes();
        for (int i = 0; i<n; i++){
            dataset.deleteWithMissing(i);
        }
        return dataset;            
    }
    
    public static Instances remove_feature(Instances dataset, int feature_num) throws Exception{
        String[] options = new String[2];
        options[0] = "-R"; // "range" 
        options[1] = String.valueOf(feature_num) + "-" + String.valueOf(feature_num);
        Remove remove = new Remove(); // new instance of filter 
        remove.setOptions(options); // set options 
        remove.setInputFormat(dataset); // inform filter about dataset // **AFTER** setting options 
        return Filter.useFilter(dataset, remove); // apply filter
} 
    
    public static void read_csv_data_to_train_test_datasets(String train_fname, String test_fname) throws IOException{
        System.out.println("---------------------------------------------------------");
        System.out.println("Reading train dataset....................................");
        System.out.println("---------------------------------------------------------");
        train_dataset = read_csv_data_source(train_fname);
        System.out.println("---------------------------------------------------------");
        System.out.println("Reading test dataset....................................");
        System.out.println("---------------------------------------------------------");
        test_dataset = read_csv_data_source(test_fname);
    }
    private static Instances read_csv_data_source(String fname_in) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(fname_in));
        return loader.getDataSet();        
    }
    
    public static void create_input_csv_files(String feature_file, String  label_file, String test_file, String train_output_file, String test_output_file) throws IOException{
        //merge feature_file and label_file into one file: output_file
        //merge in column by column manner.
        //the 2 files should have the same number of lines
            System.out.println("---------------------------------------------------------");
            System.out.println("Creating LibSVM input files (.csv) from the input files..");
            System.out.println("---------------------------------------------------------");
            
            ArrayList<String> str_test_ID = new ArrayList<String>();
            int[] test_ID; int n = 0;
            Scanner test_scan;
            test_scan = new Scanner(new File(test_file));
            while (test_scan.hasNextLine()){
                String test_line = test_scan.nextLine();
                String[] s_array = test_line.split(",");
                n = s_array.length;            
                for (int i = 0; i<n; i++) str_test_ID.add(s_array[i].trim());
            }
            test_scan.close();
            n = str_test_ID.size(); test_ID = new int[n];
            for (int i=0;i<n;i++) test_ID[i] = Integer.parseInt(str_test_ID.get(i));
            //sort
            for (int i = 0; i<n-1; i++){
                int min = test_ID[i]; int ind_min = i;        
                for (int j = i + 1; j<n; j++){
                    if (min > test_ID[j]) {min = test_ID[j]; ind_min = j;}
                }
                int tmp = test_ID[i]; test_ID[i] = min ; test_ID[ind_min] = tmp;
            }

            int line_number = 0, ind_on_test_ID = 0;
            Scanner feature_scan; Scanner label_scan; FileWriter train_fout, test_fout;
            try {
                feature_scan = new Scanner(new File(feature_file));
                label_scan = new Scanner(new File(label_file));
                train_fout = new FileWriter(train_output_file, false);
                test_fout = new FileWriter(test_output_file, false);

                if(feature_scan.hasNextLine()){
                    if (label_scan.hasNextLine()){
                        String feature_line = feature_scan.nextLine();
                        String label_line = label_scan.nextLine();                
                        String s = feature_line + "," + label_line + "\n";
                        String[] s_array = feature_line.split(",");
                        int nn = s_array.length; String header = "";
                        for (int i = 1; i<=nn; i++) header = header + "F" + String.valueOf(i) + ",";
                        header = header + "X,Y\n";
                        train_fout.write(header); test_fout.write(header);
                        if (line_number == test_ID[ind_on_test_ID]) {test_fout.write(s);ind_on_test_ID++;}
                        else train_fout.write(s);
                        line_number++;
                    }
                }            
                while (feature_scan.hasNextLine()){
                    if (label_scan.hasNextLine()){
                        String feature_line = feature_scan.nextLine();
                        String label_line = label_scan.nextLine();                
                        String s = feature_line + "," + label_line + "\n";
                        if (line_number == test_ID[ind_on_test_ID]) {test_fout.write(s);ind_on_test_ID++;}
                        else train_fout.write(s);
                        line_number++;                    
                    }
                }
                train_fout.close();
                test_fout.close();
                feature_scan.close();
                label_scan.close();
            } catch (FileNotFoundException ex) {
                Logger.getLogger(SVR.class.getName()).log(Level.SEVERE, null, ex);
            }        
    }

    private static String[] find_best_nu(Instances train_dataset, String X_options, String Y_options) throws Exception {
        String[] ret_val = new String[2];
        Instances X_train_dataset = new Instances(remove_feature(train_dataset, train_dataset.numAttributes()));
        
        int class_index = X_train_dataset.numAttributes()-1;
        //String options = "-C 2048.0 -G 0.5 -S 4 -K 2 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";
        ret_val[0] = find_best_nu_for_svc_svr(X_train_dataset, class_index, X_options);
        
        Instances Y_train_dataset = new Instances(remove_feature(train_dataset, train_dataset.numAttributes()-1));
        class_index = Y_train_dataset.numAttributes()-1;
        //options = "-C 8192.0 -G 0.5 -S 4 -K 2 -E 1.0E-4 -R 1.0E-4 -W 1 -Z";
        ret_val[1] = find_best_nu_for_svc_svr(Y_train_dataset, class_index, Y_options);
        return ret_val;
    }
}
