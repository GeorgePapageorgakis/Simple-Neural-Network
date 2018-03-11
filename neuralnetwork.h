#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <vector>

struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {
public:
	//Feedforward only need to know the #neurons of the next layer
	Neuron(unsigned numOutputs, unsigned myIndex);
	double 	getOutputVal		(void) const;
	void 	setOutputVal		(double val);
	void 	feedForward			(const Layer &prevLayer);
	void 	calcOutputGradients	(double targetVal);
	void 	calcHiddenGradients	(const Layer &nextLayer);
	void	updateInputWeights	(Layer &prevLayer);
private:
	static 	double randomWeight				 (void);
	static 	double transferFunction			 (double x);	//the sigmoid function of x that can have grad
	static 	double transferFunctionDerivative(double x);
			double sumDOW					 (const Layer &nextLayer) const;
private:
	double 	 m_outputVal;
	double 	 m_gradient;
	unsigned m_myIndex;
	static double eta;	//η [0 .. 1.0] [slow-medium-fast learner] overall net training rate
	static double alpha;//α [0 .. n] [momentum rate] multiplier of last weight change
	std::vector<Connection> m_outputWeights;//the output weights of each neuron
};

class Net {
public:
	Net(const std::vector<unsigned> &topology);
	void 	feedForward			 (const std::vector<double> &inputVals);
	void 	backProp			 (const std::vector<double> &targetVals);
	void 	getResults			 (std::vector<double> &resultVals) const;
	double 	getRecentAverageError(void) const;
private:
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
	std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
};

#endif
