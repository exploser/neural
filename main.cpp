#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <random>

using Neuron = std::vector<double>;
using LayerOutput = std::vector<double>;
using Layer = std::vector<Neuron>;
using Network = std::vector<Layer>;
using NetworkOutput = std::vector<LayerOutput>;

static double squash(double input)
{
	return 1.0 / (1.0 + std::exp(-input));
}

double getNeuronOutput(const Neuron &n, const LayerOutput &prevLayer)
{
	// input neuron
	if (prevLayer.empty())
	{
		assert(!n.empty());
		return n.front();
	}

	// bias neuron
	if (n.empty())
	{
		return 1.0;
	}

	assert(prevLayer.size() == n.size());

	// TODO: add Kahan summation?
	double result = 0;

	for (size_t i = 0; i < n.size(); ++i)
	{
		result += prevLayer[i] * n[i];
	}

	return squash(result);
}

NetworkOutput forwardPass(const Network &network)
{
	NetworkOutput result;
	LayerOutput lastLayerOutput, nextLayerOutput;

	result.reserve(network.size());

	for (auto &layer : network)
	{
		nextLayerOutput.reserve(layer.size());

		for (auto &neuron : layer)
		{
			nextLayerOutput.push_back(getNeuronOutput(neuron, lastLayerOutput));
		}

		lastLayerOutput.swap(nextLayerOutput);
		nextLayerOutput.clear();
		result.push_back(lastLayerOutput);
	}

	return result;
}

double delta(double x, double y)
{
	return x * y * (1.0 - y);
}

double learningRate = 0.5;

Network backwardPass(
	const Network &network,
	const NetworkOutput &networkResult,
	const std::vector<double> &target
	)
{
	Network newNetwork = network;

	// output layer
	size_t i = network.size() - 1;
	std::cerr << i << std::endl;
	auto layer = network[i];

	std::vector<double> nextLayerDeltas(layer.size());

	for (size_t j = 0; j < layer.size(); ++j)
	{
		auto neuron = layer[j];
		double Oj = networkResult[i][j];
		nextLayerDeltas[j] = delta(Oj - target[j], Oj);

		for (size_t k = 0; k < neuron.size(); ++k)
		{
			newNetwork[i][j][k] -= nextLayerDeltas[j] * networkResult[i - 1][k] * learningRate;
		}
	}

	// hidden layers (0 is input layer, which is ignored)
	for (--i; i > 0; --i)
	{
		auto layer = network[i];
		auto nextlayer = network[i + 1];

		std::vector<double> newDeltas(layer.size());

		for (size_t j = 0; j < layer.size(); ++j)
		{
			auto neuron = layer[j];
			double Oj = networkResult[i][j];

			double dError = 0;

			for (size_t m = 0; m < nextlayer.size(); ++m)
			{
				// delta(O_m) * weight
				dError += nextLayerDeltas[m] * nextlayer[m][j];
			}

			newDeltas[j] = delta(dError, Oj);

			for (size_t k = 0; k < neuron.size(); ++k)
			{
				newNetwork[i][j][k] -= newDeltas[j] * networkResult[i - 1][k] * learningRate;
			}
		}

		nextLayerDeltas.swap(newDeltas);
	}

	return newNetwork;
}

Network buildNetwork(
	const std::vector<size_t> &layerSizes
	)
{
	std::uniform_real_distribution<> distribution(0, 1);
	std::minstd_rand generator(std::random_device{ } ());

	Network n;
	n.reserve(layerSizes.size());

	// input layer
	n.push_back(Layer(layerSizes.front(), Neuron(1)));

	// hidden & output layers
	for (size_t i = 1; i < layerSizes.size(); ++i)
	{
		n.push_back(Layer(layerSizes[i]));

		for (size_t j = 0; j < layerSizes[i]; ++j)
		{
			n[i][j].reserve(layerSizes[i - 1]);

			for (size_t j = 0; j < layerSizes[i - 1]; ++j)
			{
				n[i][j].push_back(distribution(generator));
			}
		}
	}

	return n;
}

void setNetworkInputs(Network &n, const LayerOutput &inputs)
{
	assert(n[0].size() == inputs.size());

	for (size_t i = 0; i < inputs.size(); ++i)
	{
		n[0][i].front() = inputs[i];
	}
}

double getTotalError(const LayerOutput &networkOutput, const LayerOutput &targetOutput)
{
	assert(networkOutput.size() == targetOutput.size());

	double error = 0;

	for (size_t i = 0; i < networkOutput.size(); ++i)
	{
		error += (networkOutput[i] - targetOutput[i]) * (networkOutput[i] - targetOutput[i]);
	}
}

void dumpOutput(const LayerOutput &output)
{
	for (auto &o: output)
	{
		std::cout << o << '\t';
	}

	std::cout << std::endl;
}

int main(int argc, char *argv[])
{
	const std::vector<double> groundTruth = {0.01, 0.99};

	auto network = buildNetwork({2, 2, 2});
	setNetworkInputs(network, {0.05, 0.10});

	for (int i = 0; i < 10; ++i)
	{
		auto result = forwardPass(network);
		std::cout << "Fwd pass " << i << ", total error " << getTotalError(result.back(), groundTruth);
		network = backwardPass(network, result, groundTruth);
	}

	std::getchar();
}
