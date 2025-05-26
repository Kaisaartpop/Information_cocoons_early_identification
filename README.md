# Information_cocoons_early_identification
## 📁 Dataset Preparation

Please prepare your dataset and place it under the following directory:

```bash
../data/data_final/
```

Make sure the following files exist:

```text
data/
└── data_final/
    ├── X_train.npy     # Training data
    ├── y_train.npy     # Training labels
    ├── X_val.npy       # Validation data
    ├── y_val.npy       # Validation labels
    ├── X_test.npy      # Test data
    └── y_test.npy      # Test labels
```

Each `.npy` file should contain data with the following shapes:

```text
X_train.npy : (N, seq_length, num_nodes+1)
y_train.npy : (N,)

X_val.npy   : (N, seq_length, num_nodes+1)
y_val.npy   : (N,)

X_test.npy  : (N, seq_length, num_nodes+1)
y_test.npy  : (N,)
```

Where:
- `N` is the number of samples,
- `seq_length` is the length of the time sequence,
- `num_nodes` is the number of categories in each time step, in our work we have 35 categories of different videos.
- The '+1' dimension is the relative entropy
  

You can load the dataset in your code as follows:

```python
import numpy as np

data_train = np.load('../data/data_final/X_train.npy')
labels_train = np.load('../data/data_final/y_train.npy')

data_val = np.load('../data/data_final/X_val.npy')
labels_val = np.load('../data/data_final/y_val.npy')

data_test = np.load('../data/data_final/X_test.npy')
labels_test = np.load('../data/data_final/y_test.npy')
```
## 1️⃣ Run the Vanilla Model

To train and evaluate the vanilla model, simply run the following command:
```bash
python model.py 
```
### ⚙️ Optional Arguments

When running `model.py`, you can customize the model behavior via the following command-line arguments:

```bash
python model.py --seq_len 5 --embed_dim 64 --num_nodes 35
```

#### 📌 Argument Descriptions:

- `--seq_len` *(default: 5)*  
  Length of the input time sequence for each sample. For example, `--seq_len 5` means the model will take 5 time steps as input.

- `--embed_dim` *(default: 64)*  
  The dimension of the embedding or hidden representation used within the model. A higher value may increase model capacity but also computation.

- `--num_nodes` *(default: 35)*  
  The number of nodes or categories in your input data. 
> 💡 You can adjust these parameters to fit your specific dataset or task setting.

## 2️⃣ Ablation Study

To perform ablation studies and assess the contribution of different model components, you can run the following scripts individually:

```bash
python ablation/MLP_CNN.py               # Baseline model with CNN and MLP
python ablation/MLP.py                   # MLP only (no CNN)
python ablation/MLP_CNN_Attraction.py    # Remove node attraction modeling
python ablation/MLP_CNN_Transition.py    # Remove transition modeling
python ablation/MLP_CNN_for_S_un.py      # Baseline MLP and CNN for relative entropy only
```

Still, you can control your arguments using:

```bash
--seq_len 5 --embed_dim 64 --num_nodes 35
```

> For example:  
> `python ablation/MLP_CNN.py --seq_len 5 --embed_dim 64 --num_nodes 35`


## 3️⃣ Test the Impact of Sequence Length

To explore how different input sequence lengths affect model performance, you can prepare datasets with varying lengths and run the same model with different `--seq_len` values.

🗂️ Put your data into a directory following the format:

```
../data/data_full_{i}
```

Where `{i}` = `seq_len × 5`, e.g.:

- `../data/data_full_25` for `--seq_len 5`
- `../data/data_full_30` for `--seq_len 6`
- `../data/data_full_35` for `--seq_len 7`
- etc.

The model will automatically load data from the corresponding folder based on the `--seq_len` argument you provide.

### 🧪 Example Command

```bash
python timing/timemodel.py --seq_len 6 --embed_dim 64 --num_nodes 35
```

> 📌 This allows you to evaluate how input temporal context length impacts the final performance without changing the core code.

## 4️⃣ Visualization

To visualize model performance, comparisons, or the impact of sequence length, you can refer to the following notebook:

📍 Visualization code:  
```
/timing/plot.ipynb
```

