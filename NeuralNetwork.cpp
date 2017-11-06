#include "NeuralNetwork.hpp"
#include "NeuralNetworkLow.hpp"

using namespace neural;

struct neural::NeuralNetwork::NeuralNetworkImpl
{
	Network n;
};

NeuralNetwork::NeuralNetwork(const std::vector<size_t> &layerSizes)
	: impl(std::make_shared<NeuralNetworkImpl>())
{
    impl->n = buildNetwork(layerSizes);
}

NeuralNetwork::LayerOutput NeuralNetwork::Run(const LayerInput &input) const
{
    setNetworkInputs(impl->n, input);
    auto result = forwardPass(impl->n);
    return result.back();
}

double NeuralNetwork::Train(const LayerInput &input, const LayerOutput &groundTruth)
{
    setNetworkInputs(impl->n, input);
    auto result = forwardPass(impl->n);
    impl->n = backwardPass(impl->n, result, groundTruth);

    return getTotalError(result.back(), groundTruth);
}

double NeuralNetwork::GetTotalError(const LayerOutput &result, const LayerOutput &groundTruth)
{
	return getTotalError(result, groundTruth);
}
