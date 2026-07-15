# Continual Learning for Underwater Scenes

This repository provides implementations of continual learning methods for underwater scene segmentation.

## 1. Dataset Preparation

Place the `Underwater_CL` directory in the root directory of the repository.

The expected dataset structure is:

```text
Underwater_CL/
├── data/
│   ├── all_imgs/
│   ├── all_masks/
│   └── task_configs/
````

The dataset is loaded and processed by `dataset.py`.

## 2. Model Architecture

All model implementations are located in the `models/` directory.

The repository currently supports the following continual learning approaches:

* Naive fine-tuning
* Experience Replay (ER)
* Dark Experience Replay++ (DER++)

## 3. Training

For convenience, each method currently has a separate training script. The training pipeline may be unified and refactored in a future version to improve code organization and maintainability.

For example, run the following command to train DER++:

```bash
python derpp_train.py
```

The checkpoints, and results in JSON format will be saved in folder `checkpoints_[model name]`

Run all commands from the root directory of the repository.

## 4. Visualization

The `visualize.py` script provides a simple visualization of catastrophic forgetting across continual learning tasks.

Run the visualization script using:

```bash
python visualize.py
```
Example of the knowledge forgetting on the first Aquarium task:
<img width="6616" height="940" alt="image" src="https://github.com/user-attachments/assets/bae7fd99-0142-4807-96b5-b8885d7dfe9a" />

