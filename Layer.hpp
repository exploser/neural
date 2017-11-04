#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <cstdlib>
#include <cassert>

namespace neural
{
class Neuron;
class Layer
{
	std::vector<std::shared_ptr<Neuron>> neurons;

public:
	Layer(size_t size, std::function<std::shared_ptr<Neuron>()> generator) :
		neurons(size)
	{
		for (auto &neuron : neurons)
		{
			neuron = generator();
		}
	}

	virtual ~Layer()
	{
	}

	virtual size_t Size() const
	{
		return neurons.size();
	}

	std::shared_ptr<Neuron> GetNeuron(size_t i)
	{
		assert(i < neurons.size());
		return neurons[i];
	}

	auto GetNeurons()
	{
		return neurons;
	}
};
}
