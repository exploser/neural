#pragma once
#include <memory>
#include <vector>

namespace neural
{
class NeuralNetwork
{
	struct NeuralNetworkImpl;

	std::shared_ptr<NeuralNetworkImpl> impl;

public:
	using LayerOutput = std::vector<double>;
	using LayerInput = LayerOutput;

	NeuralNetwork(const std::vector<size_t> &layerSizes);
	LayerOutput Run(const LayerInput &input) const;
	LayerOutput GetLastOutput() const;
	double Train(const LayerInput &input, const LayerOutput &groundTruth);

	static double GetTotalError(const LayerOutput &result, const LayerOutput &groundTruth);
};
}
