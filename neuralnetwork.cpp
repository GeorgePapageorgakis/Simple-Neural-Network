/*
 * neuralnetwork.cpp
 *
 *  Created on: Jan 30, 2018
 *      Author: ngeo
 */
#include "neuralnetwork.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cassert>
//#include <exception>


Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
	for (unsigned conn = 0; conn < numOutputs; ++conn) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

double 	Neuron::getOutputVal(void) const {
	return m_outputVal;
}

void 	Neuron::setOutputVal(double val) {
	m_outputVal = val;
}

//performs summation operation, output = f( Σ_i(i_i * w_i) )
void Neuron::feedForward(const Layer &prevLayer) {
	double sum = 0.0;
	//perform a loop over all neurons from previous layer
	for (unsigned neuron = 0; neuron < prevLayer.size(); ++neuron) {
		sum += prevLayer[neuron].m_outputVal * prevLayer[neuron].m_outputWeights[m_myIndex].weight;
	}
	m_outputVal = Neuron::transferFunction(sum);
	return;
}

//normalizes output from [-1.0 .. 1.0]
double Neuron::transferFunction(double x) {
	return tanh(x);
}

//the derivative of tanh(x) (approximated)
double Neuron::transferFunctionDerivative(double x) {
	return (1.0 - (x * x));
}

void Neuron::calcOutputGradients(double targetVal) {
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
	return;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
	return;
}

double Neuron::sumDOW(const Layer &nextLayer) const {
	double sum = 0.0;
	//sum our contributions of the errors at the nodes we feed in the next layer
	for (unsigned neuron = 0; neuron < nextLayer.size() - 1; ++neuron) {
		sum += m_outputWeights[neuron].weight * nextLayer[neuron].m_gradient;
	}
	return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer) {
	//The weights to be updated are in the Connection container
	//in the neurons in the preceding layer
	for (unsigned neuronIdx = 0; neuronIdx < prevLayer.size(); ++neuronIdx) {
		Neuron &neuron = prevLayer[neuronIdx];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		//individual input, magnified by the gradient and train rate:
		//eta * neuron.getOutputVal() * m_gradient
		//also add momentum a fraction of the previous delta weight
		//+ alpha * oldDeltaWeight
		double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
	return;
}

double Neuron::randomWeight(void) {
	return rand() / static_cast<double>(RAND_MAX);
}




Net::Net(const std::vector<unsigned> &topology) : m_error(0), m_recentAverageError(0), m_recentAverageSmoothingFactor(0){
	unsigned numLayers = topology.size();

	for(unsigned layer = 0; layer < numLayers; ++layer) {
		//create a new layer
		m_layers.push_back(Layer());
		std::cout << "Neural-Net Layer #" << layer << " created..." << std::endl;

		//calculate outputs for each neuron
		unsigned numOutputs = (layer == topology.size() -1 ? 0 : topology[layer+1]);
		//fill it with i neurons and a bias neuron (+1)
		for(unsigned neuron = 0; neuron <= topology[layer]; ++neuron){
			m_layers.back().push_back(Neuron(numOutputs, neuron));
			std::cout << "\tNeuron #" << neuron << " created..." << std::endl;
		}
		//Set bias neuron output to 1.0. Last created neuron in each layer.
		m_layers.back().back().setOutputVal(1.0);
	}
}

/*
 * Assign the input values and network weights on each layer
 * */
void Net::feedForward(const std::vector<double> &inputVals) {
	//check if size of input neurons is equal to the input layer length
//	try {
		if (inputVals.size() != m_layers[0].size() - 1)
			throw std::length_error("Wrong layer length input");
//	}
//	catch(length_error le) {
//		std::cerr << "Exception thrown (length_error):" << le.what() << std::endl;
//	}
//	return EXIT_FAILURE;
	//assing the input values into the input neurons
	for(unsigned neuron = 0; neuron < inputVals.size(); ++neuron) {
		m_layers[0][neuron].setOutputVal(inputVals[neuron]);
	}

	//Propagate weights forward
	for (unsigned layer = 1;  layer < m_layers.size(); ++layer) {
		//need a reference to previous layer values
		Layer &prevLayer = m_layers[layer - 1];
		for (unsigned neuron = 0;  neuron < m_layers[layer].size() - 1; ++neuron) { //-1 for bias neuron
			m_layers[layer][neuron].feedForward(prevLayer);
		}
	}
	return;
}

/*
 * Calculate feedback optimization to the network
 * */
void Net::backProp(const std::vector<double> &targetVals) {
	//calculate overall net error (RMS of output neuron errors) RMS = sqrt((1/n) * Σi_n(target_i - actual_i)^2)
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;
	//check for all neurons in the outputLayer not including bias
	for (unsigned neuron = 0; neuron < outputLayer.size() - 1; ++neuron) {
		double delta = targetVals[neuron] - outputLayer[neuron].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1;	//get average error squared
	m_error = sqrt(m_error);

	//implement recent average measurement for convenience
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

	//calculate output layer Gradients
	for (unsigned neuron = 0; neuron < outputLayer.size() - 1; ++neuron) {
		outputLayer[neuron].calcOutputGradients(targetVals[neuron]);
	}

	//calculate Gradients of hidden layers
	for (unsigned layer = m_layers.size() - 2; layer > 0; --layer) {
		Layer &hiddenLayer = m_layers[layer];
		Layer &nextLayer = m_layers[layer + 1];

		//iterate over all neurons in hidden layer
		for (unsigned neuron = 0; neuron < hiddenLayer.size(); ++neuron) {
			hiddenLayer[neuron].calcHiddenGradients(nextLayer);
		}
	}
	//For all layers from outputs to first hidden layer update connection weights
	for (unsigned layer = m_layers.size() - 1; layer > 0; --layer) {
		Layer &currentLayer = m_layers[layer];
		Layer &prevLayer = m_layers[layer - 1];
		for (unsigned neuron = 0; neuron < currentLayer.size() - 1; ++neuron) {
			currentLayer[neuron].updateInputWeights(prevLayer);
		}
	}
	return;
}

void Net::getResults(std::vector<double> &resultVals) const{
	resultVals.clear();
	for(unsigned neuron = 0; neuron < m_layers.back().size() - 1; ++neuron){
		resultVals.push_back(m_layers.back()[neuron].getOutputVal());
	}
}

double Net::getRecentAverageError(void) const{
	return m_recentAverageError;
}
