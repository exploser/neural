#pragma once
#include <vector>
#include <memory>
#include <cstdint>

#include "Neuron.hpp"
#include "Layer.hpp"

namespace neural
{
class Network
{
	std::vector<std::shared_ptr<Layer>> layers;

public:
	Network(const std::vector<size_t> &layerSizes);
	void SetInputs(const std::vector<double> &inputs);

	std::vector<double> ForwardPass()
	{
		auto outputLayer = layers.back();
		std::vector<double> result;

		for (auto neuron : outputLayer->GetNeurons())
		{
			result.push_back(neuron->GetOutput());
		}

		return result;
	}

	std::string Dump();
};
}
