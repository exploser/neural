#pragma once
#include <memory>
#include <vector>

namespace neural
{
using Neuron = std::vector<double>;
using Layer = std::vector<Neuron>;
using LayerOutput = std::vector<double>;
using Network = std::vector<Layer>;
using NetworkOutput = std::vector<LayerOutput>;

double getNeuronOutput(const Neuron &n, const LayerOutput &prevLayer);
NetworkOutput forwardPass(const Network &network);
Network backwardPass(
	const Network &network,
	const NetworkOutput &networkResult,
	const LayerOutput &target
	);

Network buildNetwork(
	const std::vector<size_t> &layerSizes
	);

void setNetworkInputs(Network &n, const LayerOutput &inputs);

double getTotalError(const LayerOutput &networkOutput, const LayerOutput &targetOutput);
}
