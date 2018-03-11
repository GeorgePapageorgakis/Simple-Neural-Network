//============================================================================
// Name        : neural-net-tutorial.cpp
// Author      : George Papageorgakis
// Version     :
// Copyright   : Your copyright notice
// Description : Neural Network in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "neuralnetwork.h"

//Class to read training data from file
class TrainingData {
public:
	TrainingData(const std::string filename);
	bool isEof(void);
	void getTopology(std::vector<unsigned> &topology);
	//Return number of input values read from file
	unsigned getNextInputs(std::vector<double> &inputVals);
	unsigned getTargetOutputs(std::vector<double> &targetOutputVals);
private:
	std::ifstream m_trainingDataFile;
};


TrainingData::TrainingData(const std::string filename) {
	m_trainingDataFile.open(filename.c_str());
}

bool TrainingData::isEof(void) {
	return m_trainingDataFile.eof();
}

void TrainingData::getTopology(std::vector<unsigned> &topology) {
	std::string line;
	std::string label;

	getline(m_trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	if(this->isEof() || label.compare("topology:") != 0) {
		abort();
	}

	while(!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}

unsigned TrainingData::getNextInputs(std::vector<double> &inputVals) {
	inputVals.clear();
	std::string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if(label.compare("in:") == 0) {
		double oneValue;
		while(ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}
	return inputVals.size();
}
unsigned TrainingData::getTargetOutputs(std::vector<double> &targetOutputVals) {
	targetOutputVals.clear();
	std::string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if(label.compare("out:") == 0) {
		double oneValue;
		while(ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}
	return targetOutputVals.size();
}






void showVectorVals(std::string label, std::vector<double> &v) {
	std::cout << label << " ";
	for (unsigned i= 0; i < v.size(); ++i) {
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}



void generateTrainingData(void) {
    //std::ofstream outputFile;
    std::ofstream outputFile ("trainingData.txt");
    //outputFile.open("trainingData.txt");
    std::cout<< "Creating file and opening it for writing..." << std::endl;

    //random training sets for XOR - 2 in 1 out
    //Can be trained with more complex Boolean Expressions
    outputFile << "topology: 2 4 1" << std::endl;
    for (int i = 2000; i >= 0; --i) {
        int n1 = static_cast<int>(2.0 * rand() / static_cast<double>(RAND_MAX));
        int n2 = static_cast<int>(2.0 * rand() / static_cast<double>(RAND_MAX));
        int t = n1 ^ n2;    //0 or 1
        outputFile << "in: " << n1 << ".0 " << n2 << ".0 " << std::endl;
        outputFile << "out: " << t << ".0 " << std::endl;
    }
    outputFile.close();
    return;
}

int main(void) {

	std::cout << "Neural Network Tutorial" << std::endl;
	generateTrainingData();
	TrainingData trainData("trainingData.txt");

	//example { 3, 2, 1 }
	std::vector<unsigned> topology;
	//	topology.push_back(3);
	//	topology.push_back(2);
	//	topology.push_back(1);
	trainData.getTopology(topology);
	Net myNet(topology);

	std::vector<double> inputVals, targetVals, resultVals;
//	myNet.feedForward(inputVals);
//	myNet.backProp(targetVals);
//	myNet.getResults(resultVals);

	int trainingPass = 0;
	while(!trainData.isEof()){
		++trainingPass;
		std::cout << "Pass " << trainingPass;
		//Get new input data and feed it forward
		if(trainData.getNextInputs(inputVals) != topology[0]){
			break;
		}
		showVectorVals(": Inputs: ", inputVals);
		myNet.feedForward(inputVals);

		//collect the net's actual results
		myNet.getResults(resultVals);
		showVectorVals("Outputs: ", resultVals);

		//train the net what the outputs should have been
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets: ", targetVals);
		//assert(targetVals.size() == topology.back());
		myNet.backProp(targetVals);

		//Report neural net learning performance
		std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl;
	}
	std::cout << "Done!" << std::endl;

	return 0;
}
