#pragma once
#include <memory>
#include <algorithm>

#include "Neuron.hpp"
#include "Layer.hpp"
#include "Random.hpp"

namespace neural
{
class HiddenNeuron :
	public Neuron
{
	std::vector<double> weights;
	std::weak_ptr<Layer> prevLayer;

public:
	HiddenNeuron(const std::shared_ptr<Layer> &prevLayer)
		: prevLayer(prevLayer), weights(prevLayer->Size() + 1)
	{
		std::uniform_real_distribution<> distribution(0, 1);
		std::generate(
			weights.begin(),
			weights.end(),
			[&] {
				return distribution(Random::Generator());
			}
			);
	}

	virtual double GetOutput() const override
	{
		double result = 0;
		auto layer = prevLayer.lock();

		assert(weights.size() == layer->Size() + 1);

		for (size_t i = 0; i < layer->Size(); ++i)
		{
			result += weights[i] * layer->GetNeuron(i)->GetOutput();
		}

		// return result + bias
		return squash(result + weights.back());
	}

private:
	static double squash(double input)
	{
		return 1.0 / (1.0 + std::exp(-input));
	}
};
}
