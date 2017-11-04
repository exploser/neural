#include "NeuralNet.hpp"

#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <sstream>
#include <iomanip>

#include "Neuron.hpp"
#include "InputNeuron.hpp"
#include "HiddenNeuron.hpp"
#include "OutputNeuron.hpp"

using namespace neural;

double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x / 2.0));
}

constexpr double signalMultiplier = 0.1;

Network::Network(const std::vector<size_t> &layerSizes)
{
	assert(layerSizes.size() > 1);

	auto it = layerSizes.begin();

	layers.emplace_back(
		std::make_shared<Layer>(
			*(it++),
			[] {
				return std::make_shared<InputNeuron>();
			}
			)
		);

	while (it != layerSizes.end() - 1)
	{
		layers.emplace_back(
			std::make_shared<Layer>(
				*(it++),
				[this] {
					return std::make_shared<HiddenNeuron>(layers.back());
				}
				)
			);
	}

	layers.emplace_back(
		std::make_shared<Layer>(
			*(it++),
			[this] {
				return std::make_shared<OutputNeuron>(layers.back());
			}
			)
		);
}

void Network::SetInputs(const std::vector<double> &inputs)
{
	auto inputLayer = layers[0];

	for (int i = 0; i < inputs.size(); ++i)
	{
		std::dynamic_pointer_cast<InputNeuron>(inputLayer->GetNeuron(i))->Set(inputs[i]);
	}
}

std::string Network::Dump()
{
	std::stringstream ss;
	ss << std::fixed << std::setprecision(5);

	for (auto &layer : layers)
	{
		for (auto &neuron : layer->GetNeurons())
		{
			ss << neuron->GetOutput() << '\t';
		}

		ss << std::endl;
	}

	return ss.str();
}
