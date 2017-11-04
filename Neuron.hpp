#pragma once

namespace neural
{
struct Neuron
{
	virtual double GetOutput() const = 0;
};
}
