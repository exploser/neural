#pragma once
#include "Neuron.hpp"

namespace neural
{
class InputNeuron :
	public Neuron
{
	double input = 0;

public:

	void Set(double input)
	{
		this->input = input;
	}

	virtual double GetOutput() const override
	{
		return input;
	}
};
}
