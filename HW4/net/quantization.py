import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix

import torch


def apply_weight_sharing(model, bits=5):
    """
    Applies weight sharing to the given model
    """
    for name, module in model.named_children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        quan_range = 2 ** bits
        if len(shape) == 2:  # Fully connected layers
            print(f'{name:20} | {str(module.weight.size()):35} | => Quantize to {quan_range} indices')
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

            # Weight sharing by kmeans
            space = np.linspace(min(mat.data), max(mat.data), num=quan_range)
            kmeans = KMeans(
                n_clusters=len(space),
                init=space.reshape(-1, 1),
                n_init=1,
                algorithm="lloyd"
            )
            kmeans.fit(mat.data.reshape(-1, 1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight

            # Insert to model
            module.weight.data = torch.from_numpy(mat.toarray()).to(dev)
        elif len(shape) == 4:  # Convolution layers

            #################################
            # TODO:
            #    Suppose the weights of a certain convolution layer are called "W"
            #       1. Get the unpruned (non-zero) weights, "non-zero-W",  from "W"
            #       2. Use KMeans algorithm to cluster "non-zero-W" to (2 ** bits) categories
            #       3. For weights belonging to a certain category, replace their weights with the centroid
            #          value of that category
            #       4. Save the replaced weights in "module.weight.data", and need to make sure their indices
            #          are consistent with the original
            #   Finally, the weights of a certain convolution layer will only be composed of (2 ** bits) float numbers
            #   and zero
            #   --------------------------------------------------------
            #   In addition, there is no need to return in this function ("model" can be considered as call by
            #   reference)
            #################################

            print(f'{name:20} | {str(module.weight.size()):35} | ** NEED TO BE IMPLEMENTED **')
            # 1. Get the unpruned (non-zero) weights and their indices
            mask = (weight != 0)
            non_zero_weights = weight[mask] # Get the values where the mask is True
            # Get the indices of the non-zero elements. np.where returns a tuple of arrays, one for each dimension (out_c, in_c, k_h, k_w).
            non_zero_indices = np.where(mask)

            # Handle case where the layer might be fully pruned or has no non-zero weights (e.g., after severe pruning)
            if non_zero_weights.size == 0:
                print(f'{name:20} | No non-zero weights to quantize.')
                # If there were originally non-zeros that were pruned, they should already be zero.
                # If the original tensor was all zeros, it stays all zeros.
                # We can just skip the clustering and replacement for this layer.
                continue # Move to the next module


            # 2. Use KMeans algorithm to cluster "non-zero-W"
            # Prepare data for KMeans: it expects a 2D array
            non_zero_weights_reshaped = non_zero_weights.reshape(-1, 1)

            # Initialize centroids evenly across the range of non-zero weights, similar to FC layers
            # Handle case where min == max for non-zero values
            min_val = np.min(non_zero_weights)
            max_val = np.max(non_zero_weights)

            if min_val == max_val:
                print(f'{name:20} | All non-zero weights are the same value ({min_val}). No clustering needed.')
                # Create a new array with the original shape, set non-zeros to this single value
                new_weight_array = np.zeros_like(weight)
                # Find original non-zero positions using the mask and set them to this value
                new_weight_array[mask] = min_val
                # The single unique value is just min_val, effectively 1 cluster.
                quantized_non_zero_values = np.array([min_val]) # Represent the single unique value

            else:
                # Normal KMeans clustering
                # Initialize centroids spread across the range of non-zero values
                space = np.linspace(min_val, max_val, num=quan_range)
                # Ensure init has the correct shape (num_clusters, num_features) -> (quan_range, 1)
                init_centroids = space.reshape(-1, 1)

                # KMeans handles n_clusters > n_samples or n_clusters > unique samples by capping n_clusters
                # No need for manual check for n_clusters_actual unless you want a warning.
                # unique_non_zero_count = len(np.unique(non_zero_weights))
                # n_clusters_actual = min(quan_range, unique_non_zero_count)
                # if n_clusters_actual < quan_range:
                #      print(f'{name:20} | Warning: Unique non-zero weights ({unique_non_zero_count}) less than {quan_range} clusters. Using {n_clusters_actual} clusters.')
                #      pass # Let KMeans handle it

                kmeans = KMeans(
                    n_clusters=quan_range, # KMeans handles n_clusters > n_samples
                    init=init_centroids,
                    n_init=1, # Use n_init=1 as per FC layer implementation
                    algorithm="lloyd"
                )
                # Fit KMeans on the reshaped non-zero weights
                kmeans.fit(non_zero_weights_reshaped)

                # 3. For weights belonging to a certain category, replace their weights with the centroid value
                # Get the cluster label for each non-zero weight
                labels = kmeans.labels_
                # Get the centroid values (flattened to 1D array)
                centroids = kmeans.cluster_centers_.flatten()
                # Map each label back to its corresponding centroid value
                quantized_non_zero_values = centroids[labels] # This is a 1D array matching the order of non_zero_weights


            # 4. Save the replaced weights in "module.weight.data", ensuring indices are consistent
            # Create a new weight array of the original shape, initialized to zeros
            new_weight_array = np.zeros_like(weight)

            # Place the quantized non-zero values back into their original positions
            # np.where(mask) returns a tuple of arrays, each array containing the indices along a specific dimension.
            # We can use this tuple directly for indexing the new_weight_array
            new_weight_array[non_zero_indices] = quantized_non_zero_values

            # Insert the new weight array back into the module's weight data
            module.weight.data = torch.from_numpy(new_weight_array).to(dev)

            # Optional: Verify that the number of unique non-zero values is <= quan_range + 1 (if 0 is counted)
            # Get the mask again from the updated weights to correctly count non-zeros
            updated_mask = (module.weight.data.cpu().numpy() != 0)
            unique_values_after_quant = np.unique(module.weight.data.cpu().numpy()[updated_mask])
            print(f'{name:20} | After quantization: {len(unique_values_after_quant)} unique non-zero values.')


        # Bias terms are typically not quantized in this manner, the loop iterates over modules
        # which contain weights and biases. We are only modifying module.weight.data.

