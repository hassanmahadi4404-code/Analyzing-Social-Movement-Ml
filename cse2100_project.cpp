#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>
using namespace std;

// Data structure
class MovementData {
public:
    double A, B, C, D, E, F, G;  // 7 input factors (1-5 scale)
    double H;                     // Output factor (Overall Impact, 1-10 scale)
    double successScore;          // Normalized H (0-1)
};

// Neural Network Class
class MovementBrain {
private:
    // Network weights - 7 inputs, 7 hidden neurons
    double inputToHidden[7][7];
    double hiddenBias[7];
    double hiddenToOutput[7];
    double outputBias;
    
    // Data
    vector<MovementData> trainingData;
    vector<MovementData> testingData;
    
    // Activation function
    double activate(double x) {
        return 1.0 / (1.0 + exp(-x));
    }
    
    double activateDerivative(double x) {
        double s = activate(x);
        return s * (1.0 - s);
    }
    
    // READ CSV FILE FUNCTION - handles timestamp and H on 1-10 scale
    vector<MovementData> readCSVData(string filePath) {
        vector<MovementData> allData;
        ifstream file(filePath);
        
        if (!file.is_open()) {
            cout << "ERROR: Cannot open file: " << filePath << endl;
            cout << "Please check if file exists!" << endl;
            return allData;
        }
        
        string line;
        int lineNumber = 0;
        
        cout << "Reading CSV file: " << filePath << endl;
        
        while (getline(file, line)) {
            lineNumber++;
            if (lineNumber == 1) continue; // skip header
            if (line.empty()) continue;
            
            MovementData newData;
            stringstream ss(line);
            string cell;
            int column = 0;
            
            // Determine separator
            size_t tabCount = count(line.begin(), line.end(), '\t');
            size_t commaCount = count(line.begin(), line.end(), ',');
            char separator = (tabCount > commaCount) ? '\t' : ',';
            
            while (getline(ss, cell, separator)) {
                column++;
                
                // Skip timestamp column (first column)
                if (column == 1) continue;
                
                try {
                    double value = stod(cell);
                    
                    // Columns: 2->A, 3->B, 4->C, 5->D, 6->E, 7->F, 8->G, 9->H
                    if (column == 2) newData.A = value / 5.0;          // normalize A-G to 0-1
                    else if (column == 3) newData.B = value / 5.0;
                    else if (column == 4) newData.C = value / 5.0;
                    else if (column == 5) newData.D = value / 5.0;
                    else if (column == 6) newData.E = value / 5.0;
                    else if (column == 7) newData.F = value / 5.0;
                    else if (column == 8) newData.G = value / 5.0;
                    else if (column == 9) {
                        newData.H = value;                             // H is on 1-10 scale
                        // Map 1-10 to 0-1
                        newData.successScore = (value - 1.0) / 9.0;
                    }
                }
                catch (...) {
                    break; // skip bad data
                }
            }
            
            // Only add if we have all 8 data columns (A-G + H)
            if (column >= 9) {
                allData.push_back(newData);
            }
        }
        
        file.close();
        cout << "Loaded " << allData.size() << " records from CSV" << endl;
        return allData;
    }
    
    // INITIALIZE NETWORK WEIGHTS
    void initializeBrain() {
        srand(static_cast<unsigned int>(time(0)));
        
        for (int i = 0; i < 7; i++)
            for (int j = 0; j < 7; j++)
                inputToHidden[i][j] = (rand() % 1000) / 1000.0 - 0.5;
        
        for (int i = 0; i < 7; i++)
            hiddenBias[i] = (rand() % 1000) / 1000.0 - 0.5;
        
        for (int i = 0; i < 7; i++)
            hiddenToOutput[i] = (rand() % 1000) / 1000.0 - 0.5;
        
        outputBias = (rand() % 1000) / 1000.0 - 0.5;
        
        cout << "Brain initialized with random weights (7x7 network)." << endl;
    }
    
    // SPLIT DATA INTO TRAIN AND TEST
    void splitData(vector<MovementData> allData, double trainRatio = 0.8) {
        //random_shuffle(allData.begin(), allData.end());
        
        int trainSize = allData.size() * trainRatio;
        for (int i = 0; i < allData.size(); i++) {
            if (i < trainSize)
                trainingData.push_back(allData[i]);
            else
                testingData.push_back(allData[i]);
        }
        
        cout << "Training data: " << trainingData.size() << " samples" << endl;
        cout << "Testing data: " << testingData.size() << " samples" << endl;
    }

public:
    // CONSTRUCTOR
    MovementBrain(string filePath) {
        cout << "\n==========================================" << endl;
        cout << "   SOCIAL MOVEMENT ANALYZER" << endl;
        cout << "==========================================" << endl;
        
        vector<MovementData> allData = readCSVData(filePath);
        if (allData.empty()) {
            cout << "ERROR: No data loaded. Program will exit." << endl;
            exit(1);
        }
        
        splitData(allData, 0.8);
        initializeBrain();
        cout << "Neural network ready!" << endl;
    }
    
    // MAKE PREDICTION (inputs are raw 1-5 values)
    double predict(double a, double b, double c, double d,
                   double e, double f, double g) {
        double inputs[7] = { a/5.0, b/5.0, c/5.0, d/5.0,
                             e/5.0, f/5.0, g/5.0 };
        
        double hidden[7] = {0};
        for (int j = 0; j < 7; j++) {
            double sum = hiddenBias[j];
            for (int i = 0; i < 7; i++)
                sum += inputs[i] * inputToHidden[i][j];
            hidden[j] = activate(sum);
        }
        
        double output = outputBias;
        for (int i = 0; i < 7; i++)
            output += hidden[i] * hiddenToOutput[i];
        
        return activate(output);   // returns successScore (0-1)
    }
    
    // TRAIN THE NETWORK
    void trainBrain(int cycles = 500, double learnRate = 0.1) {
        cout << "\n==========================================" << endl;
        cout << "   TRAINING STARTED" << endl;
        cout << "==========================================" << endl;
        cout << "Training cycles: " << cycles << endl;
        cout << "Learning rate: " << learnRate << endl;
        
        for (int cycle = 0; cycle < cycles; cycle++) {
            double totalError = 0.0;
            
            for (MovementData data : trainingData) {
                double inputs[7] = { data.A, data.B, data.C, data.D,
                                     data.E, data.F, data.G };
                
                double hidden[7] = {0};
                double hiddenSum[7] = {0};
                
                // Forward pass - hidden layer
                for (int j = 0; j < 7; j++) {
                    double sum = hiddenBias[j];
                    for (int i = 0; i < 7; i++)
                        sum += inputs[i] * inputToHidden[i][j];
                    hiddenSum[j] = sum;
                    hidden[j] = activate(sum);
                }
                
                // Forward pass - output layer
                double outputSum = outputBias;
                for (int i = 0; i < 7; i++)
                    outputSum += hidden[i] * hiddenToOutput[i];
                double prediction = activate(outputSum);
                
                // Calculate error (MSE)
                double error = data.successScore - prediction;
                totalError += error * error;
                
                // Backpropagation - output layer
                double outputDelta = error * activateDerivative(prediction);
                
                // Update hidden->output weights
                for (int i = 0; i < 7; i++)
                    hiddenToOutput[i] += learnRate * outputDelta * hidden[i];
                outputBias += learnRate * outputDelta;
                
                // Backpropagation - hidden layer
                double hiddenDelta[7] = {0};
                for (int i = 0; i < 7; i++) {
                    double hiddenError = outputDelta * hiddenToOutput[i];
                    hiddenDelta[i] = hiddenError * activateDerivative(hidden[i]);
                }
                
                // Update input->hidden weights
                for (int i = 0; i < 7; i++)
                    for (int j = 0; j < 7; j++)
                        inputToHidden[i][j] += learnRate * hiddenDelta[j] * inputs[i];
                
                // Update hidden biases
                for (int i = 0; i < 7; i++)
                    hiddenBias[i] += learnRate * hiddenDelta[i];
            }
            
            if (cycle % 50 == 0)
                cout << "Cycle " << cycle << " - Error: " << totalError << endl;
        }
        
        cout << "Training completed!" << endl;
    }
    
    // TEST THE NETWORK
    void testBrain() {
        cout << "\n==========================================" << endl;
        cout << "   TESTING RESULTS" << endl;
        cout << "==========================================" << endl;
        
        double totalError = 0.0;
        int correctCount = 0;
        
        for (int i = 0; i < testingData.size(); i++) {
            MovementData data = testingData[i];
            double prediction = predict(data.A*5, data.B*5, data.C*5,
                                        data.D*5, data.E*5, data.F*5, data.G*5);
            double actual = data.successScore;
            double error = abs(prediction - actual);
            totalError += error;
            
            // Convert prediction back to H (1-10 scale)
            double predictedH = (prediction * 9.0) + 1.0;
            double actualH = data.H;
            
            // Consider correct if within 1.0 on 1-10 scale
            if (abs(predictedH - actualH) <= 1.0)
                correctCount++;
            
            cout << "Test " << (i+1) << ": "
                 << "Predicted H=" << predictedH
                 << " Actual H=" << actualH
                 << " Error=" << error << endl;
        }
        
        double avgError = totalError / testingData.size();
        double accuracy = (double)correctCount / testingData.size() * 100;
        
        cout << "\n==========================================" << endl;
        cout << "SUMMARY:" << endl;
        cout << "Average Error: " << avgError << endl;
        cout << "Accuracy: " << accuracy << "%" << endl;
        cout << "Correct: " << correctCount << "/" << testingData.size() << endl;
        cout << "==========================================" << endl;
    }
    
    string getCategory(double score) {
        if (score >= 0.8) return "Excellent";
        else if (score >= 0.6) return "Good";
        else if (score >= 0.4) return "Average";
        else if (score >= 0.2) return "Poor";
        else return "Very Poor";
    }
    
    // ANALYZE FEATURE IMPORTANCE - no visual bars
    void analyzeFeatures() {
        cout << "\n==========================================" << endl;
        cout << "   FEATURE IMPORTANCE" << endl;
        cout << "==========================================" << endl;
        
        string featureNames[7] = { "Public Support (A)", "Non-violence (B)",
                                   "Inclusiveness (C)", "Organization (D)",
                                   "Damage (E)", "Instability (F)",
                                   "Misinformation (G)" };
        
        for (int f = 0; f < 7; f++) {
            double importance = 0.0;
            for (int h = 0; h < 7; h++)
                importance += abs(inputToHidden[f][h]);
            importance /= 7.0;
            
            cout << featureNames[f] << ": " << importance << endl;
        }
    }
    
    // PREDICT NEW MOVEMENT (interactive)
    void predictNewMovement() {
        cout << "\n==========================================" << endl;
        cout << "   ANALYZE NEW MOVEMENT" << endl;
        cout << "==========================================" << endl;
        cout << "Enter values (1-5) for each factor:" << endl;
        
        double a, b, c, d, e, f, g;
        
        cout << "\nA. Level of public support for the movement:" << endl;
        cout << "1) Very low\n2) Low\n3) Moderate\n4) High\n5) Very high\nEnter: ";
        cin >> a;
        
        cout << "\nB. Level of violence / non-violence:" << endl;
        cout << "1) Very violent\n2) Violent\n3) Neutral\n4) Mostly non-violent\n5) Completely non-violent\nEnter: ";
        cin >> b;
        
        cout << "\nC. Inclusiveness of the movement:" << endl;
        cout << "1) Not inclusive\n2) Slightly inclusive\n3) Neutral\n4) Inclusive\n5) Highly inclusive\nEnter: ";
        cin >> c;
        
        cout << "\nD. Clarity and organization:" << endl;
        cout << "1) Very unclear\n2) Unclear\n3) Average\n4) Clear\n5) Very clear\nEnter: ";
        cin >> d;
        
        cout << "\nE. Level of property/economic damage:" << endl;
        cout << "1) No damage\n2) Minor damage\n3) Moderate damage\n4) High damage\n5) Severe damage\nEnter: ";
        cin >> e;
        
        cout << "\nF. Social or political instability caused:" << endl;
        cout << "1) No instability\n2) Low\n3) Moderate\n4) High\n5) Extreme\nEnter: ";
        cin >> f;
        
        cout << "\nG. Spread of misinformation or rumors:" << endl;
        cout << "1) None\n2) Very little\n3) Moderate\n4) High\n5) Very high\nEnter: ";
        cin >> g;
        
        // Clamp to 1-5
        a = max(1.0, min(5.0, a));
        b = max(1.0, min(5.0, b));
        c = max(1.0, min(5.0, c));
        d = max(1.0, min(5.0, d));
        e = max(1.0, min(5.0, e));
        f = max(1.0, min(5.0, f));
        g = max(1.0, min(5.0, g));
        
        double prediction = predict(a, b, c, d, e, f, g);
        double hScore = (prediction * 9.0) + 1.0;   // Convert back to 1-10 scale
        string impactCategory = getImpactCategory(hScore);
        
        cout << "\n==========================================" << endl;
        cout << "   PREDICTION FOR H (Overall Impact)" << endl;
        cout << "==========================================" << endl;
        cout << "Predicted H Score: " << hScore << " / 10" << endl;
        cout << "Impact Category: " << impactCategory << endl;
        cout << "Success Probability: " << (prediction * 100) << "%" << endl;
        
        cout << "\n==========================================" << endl;
        cout << "   RECOMMENDATIONS" << endl;
        cout << "==========================================" << endl;
        
        if (a < 3) cout << "• Work on increasing public support" << endl;
        if (b < 3) cout << "• Focus on non-violent methods" << endl;
        if (c < 3) cout << "• Make movement more inclusive" << endl;
        if (d < 3) cout << "• Improve organization" << endl;
        if (e > 3) cout << "• Reduce economic damage" << endl;
        if (f > 3) cout << "• Decrease social instability" << endl;
        if (g > 3) cout << "• Counter misinformation" << endl;
        
        if (prediction * 100 >= 70)
            cout << "\nExcellent potential for success!" << endl;
        else if (prediction * 100 >= 50)
            cout << "\nGood potential, needs some improvements" << endl;
        else
            cout << "\nNeeds significant improvements" << endl;
    }
    
    string getImpactCategory(double hScore) {
        if (hScore >= 8.5) return "Very Positive";
        else if (hScore >= 6.5) return "Positive";
        else if (hScore >= 4.5) return "Neutral";
        else if (hScore >= 2.5) return "Negative";
        else return "Very Negative";
    }
};

// MAIN FUNCTION
int main() {
    cout << "==========================================" << endl;
    cout << "   SOCIAL MOVEMENT SUCCESS PREDICTOR" << endl;
    cout << "   Version 2.0 (H scale 1-10)" << endl;
    cout << "==========================================" << endl;
    
    // Use the exact path provided
    string csvPath = "/Users/mahadi/Desktop/Merged_Movements_final.csv";
    
    try {
        MovementBrain brain(csvPath);
        brain.trainBrain(300, 0.1);
        brain.testBrain();
        brain.analyzeFeatures();
        
        char choice;
        do {
            brain.predictNewMovement();
            cout << "\nAnalyze another movement? (y/n): ";
            cin >> choice;
            cin.ignore();
        } while (choice == 'y' || choice == 'Y');
        
        cout << "\n==========================================" << endl;
        cout << "   PROGRAM FINISHED" << endl;
        cout << "==========================================" << endl;
    }
    catch (...) {
        cout << "\nERROR! Something went wrong." << endl;
        cout << "Check CSV file and format." << endl;
    }
    
    cout << "\nPress Enter to exit...";
    cin.get();
    return 0;
}