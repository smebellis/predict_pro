train Module
============

.. py:function:: load_tensor(file_path: str) -> torch.Tensor
    
    Load a tensor from a given file path.

    Args:
        file_path (str): Path to the tensor file.

    Returns:
        torch.Tensor: Loaded tensor.

.. py:function:: evaluate(model, dataloader, criterion, device)

    Evaluate the model on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[float, float]: Average loss and accuracy.

.. py:function:: get_class_names(label_encoder_path: str, unique_labels: np.ndarray) -> list

    Retrieve class names from a LabelEncoder if available, else use label indices.

    Args:
        label_encoder_path (str): Path to the saved LabelEncoder.
        unique_labels (np.ndarray): Array of unique label indices.

    Returns:
        list: List of class names.