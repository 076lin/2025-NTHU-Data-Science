import numpy as np
from torch.nn.modules.module import Module
from torch import abs,quantile,nn,std

class PruningModule(Module):
    DEFAULT_PRUNE_RATE = {
        'conv1': 84,
        'conv2': 38,
        'conv3': 35,
        'conv4': 37,
        'conv5': 37,
        'fc1': 9,
        'fc2': 9,
        'fc3': 25
    }

    def _prune(self, module, threshold):

        #################################
        # TODO:
        #    1. Use "module.weight.data" to get the weights of a certain layer of the model
        #    2. Set weights whose absolute value is less than threshold to 0, and keep the rest unchanged
        #    3. Save the results of the step 2 back to "module.weight.data"
        #    --------------------------------------------------------
        #    In addition, there is no need to return in this function ("module" can be considered as call by
        #    reference)
        #################################
        weights = module.weight.data
        mask = abs(weights) >= threshold
        weights.mul_(mask.float())

    def prune_by_percentile(self, q=DEFAULT_PRUNE_RATE):

        ########################
        # TODO
        # 	For each layer of weights W (including fc and conv layers) in the model, obtain the (100 - q)th percentile
        # 	of absolute W as the threshold, and then set the absolute weights less than threshold to 0 , and the rest
        # 	remain unchanged.
        ########################

        # Calculate percentile value
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and (name in q):
                weights = module.weight.data

                prune_percent = 100 - q[name]
                if prune_percent < 0: prune_percent = 0
                elif prune_percent > 100: prune_percent = 100
                quantile_value = prune_percent / 100.0
                if weights.is_cuda:
                    threshold = quantile(abs(weights), quantile_value)
                    threshold_scalar = threshold.item()
                else:
                    weights_np = weights.cpu().numpy().flatten()
                    threshold_np = np.percentile(np.abs(weights_np), prune_percent)
                    threshold_scalar = float(threshold_np)
                self._prune(module, threshold_scalar)

    def prune_by_std(self, s=0.25):
        for name, module in self.named_modules():

            #################################
            # TODO:
            #    Only fully connected layers were considered, but convolution layers also needed
            #################################

            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weights = module.weight.data
                if weights.is_cuda:
                    threshold = std(weights) * s
                    threshold_scalar = threshold.item()
                else:
                    weights_np = weights.cpu().numpy()
                    threshold_np = np.std(weights_np) * s
                    threshold_scalar = float(threshold_np)

                self._prune(module, threshold_scalar)
