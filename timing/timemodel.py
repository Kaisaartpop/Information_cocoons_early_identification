import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch.utils.data import Dataset, DataLoader, random_split
import copy
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
filters=[64,64,8]
kernels=[3,2,1]
paddings = [0, 0, 0]
# Step 1: Load data
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=5, help='seq len')
parser.add_argument('--embed_dim', type=int, default=64, help='embed dim')
parser.add_argument('--num_nodes', type=int, default=35, help='categories')
args = parser.parse_args()
i=args.seq_len*5
data_train=np.load(f'../data/data_full_{i}/X_train.npy')
labels_train=np.load(f'../data/data_full_{i}/y_train.npy')
data_val=np.load(f'../data/data_full_{i}/X_val.npy')
labels_val=np.load(f'../data/data_full_{i}/y_val.npy')
data_test=np.load(f'../data/data_full_{i}/X_test.npy')
labels_test=np.load(f'../data/data_full_{i}/y_test.npy')

# Step 3: Define PyTorch Dataset class
print(data_train.shape)
class CustomDataset(Dataset):
    def __init__(self, data_temporal, labels):
        data_temporal=np.transpose(data_temporal, axes=(0, 2, 1))
        self.data_temporal = torch.tensor(data_temporal, dtype=torch.float32)
        #self.data_static = torch.tensor(data_static, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_temporal = self.data_temporal[idx]
        
        y = self.labels[idx]
        return x_temporal, y

# Create dataset objects
train_dataset = CustomDataset(data_train,  labels_train)
val_dataset = CustomDataset(data_val ,labels_val)
test_dataset = CustomDataset(data_test, labels_test)
# Create data loaders
batch_size =2048
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


embed_ids=[119,120,121,122,123,1]+list(range(76,84))
feature_sizes=[3,8,5,7,6]+[1]*(len(embed_ids)-5)
num_x_ids=list(range(6,41))
num_y_ids=list(range(41,76))
active_ids=list(range(84,119))

lis=[num_x_ids,num_y_ids,active_ids]

    
import torch
import torch.nn as nn

class Cocoons_prediction(nn.Module):
    def __init__(self, seq_len=5, embed_dim=64, num_nodes=35):
        super(Cocoons_prediction, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.total_channels = num_nodes + 1  

        self.channels = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=embed_dim, kernel_size=2, padding=1),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.AdaptiveAvgPool1d(1), 
                nn.Flatten()
            )
            for _ in range(self.total_channels)
        ])

        self.addict = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(num_nodes)
        ])

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 3, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            for _ in range(num_nodes)
        ])
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + self.total_channels * self.seq_len + num_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.adj = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))

        
        self.W_q = nn.Parameter(torch.FloatTensor(num_nodes, embed_dim, embed_dim))
        self.W_k = nn.Parameter(torch.FloatTensor(num_nodes, embed_dim, embed_dim))
        self.b_q = nn.Parameter(torch.FloatTensor(num_nodes))
        self.b_k = nn.Parameter(torch.FloatTensor(num_nodes))

        self.W_v = nn.Linear(embed_dim, embed_dim)  

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim * 3, num_heads=1)

       
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.adj)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.zeros_(self.b_q)
        nn.init.zeros_(self.b_k)

    def forward(self, x_temporal):
        x_temporal = x_temporal.to(self.adj.device)
        x_temporal[:, 1:, :] = x_temporal[:, 1:, :] / 18000
        N = x_temporal.shape[0]

        # (N, total_channels=36, seq_len) -> each (N, 1, seq_len)
        channel_outputs = [
            channel(x_temporal[:, i, :].unsqueeze(1)) 
            for i, channel in enumerate(self.channels)
        ]
        channel_outputs = torch.stack(channel_outputs, dim=1)  # (N, total_channels, embed_dim)

        # (N, num_nodes, embed_dim)
        cat_outputs = [
            layer(channel_outputs[:, i + 1, :]) 
            for i, layer in enumerate(self.addict)
        ]
        cat_outputs = torch.stack(cat_outputs, dim=1)
        cat_outputs1 = cat_outputs.detach().clone()

        # QK attention
        Q = torch.einsum('bni,nii->bni', cat_outputs, self.W_q) + self.b_q.unsqueeze(0).unsqueeze(-1)
        K = torch.einsum('bni,nii->bni', cat_outputs, self.W_k) + self.b_k.unsqueeze(0).unsqueeze(-1)

        Q_norm = torch.norm(Q, p=2, dim=-1, keepdim=True)
        K_norm = torch.norm(K, p=2, dim=-1, keepdim=True)
        cosine_similarity = (torch.matmul(Q, K.transpose(-2, -1))) / (Q_norm * K_norm.transpose(-2, -1) + 1e-8)

        
        # cat_outputs = torch.matmul(cosine_similarity, cat_outputs)
        # adj = torch.eye(self.num_nodes).to(self.adj.device) + F.softmax(F.relu(self.adj), dim=1)
        # cat_outputs = torch.matmul(adj, cat_outputs)
        cosine_similarity+= self.adj
        cat_outputs = torch.matmul(F.softmax(cosine_similarity,dim=2), cat_outputs)
        
        total_outputs = torch.cat([
            channel_outputs[:, 1:, :],  # (N, num_nodes, embed_dim)
            cat_outputs1,               # (N, num_nodes, embed_dim)
            cat_outputs                 # (N, num_nodes, embed_dim)
        ], dim=2)  # => (N, num_nodes, embed_dim * 3)

        # decode
        decode = [
            decoder(total_outputs[:, i, :]) 
            for i, decoder in enumerate(self.decoder)
        ]
        decode = torch.cat(decode, dim=1)  # (N, num_nodes)

        xc = self.fc(torch.cat([
            channel_outputs[:, 0, :],         
            x_temporal.view(N, -1),           
            decode                            
        ], dim=1))

        output = torch.sigmoid(xc)
        return output

model =Cocoons_prediction(seq_len=args.seq_len, embed_dim=args.embed_dim, num_nodes=args.num_nodes).to(device)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.to(device)
criterion = nn.BCELoss()
thrs=0.33
#criterion=F.binary_cross_entropy
optimizer = optim.Adam(model.parameters(), lr=0.0008)
# Training loop
num_epochs = 20
best_auc=0
for epoch in range(num_epochs):
    model.train()
    train_loss=0
    
    for x_temporal, y in train_loader:
        y=y.to(device)
        optimizer.zero_grad()
        outputs = model(x_temporal)
        # print(f'outputs device: {outputs.device}')
        # print(f'y device: {y.device}')
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss/=len(train_loader)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

    model.eval()
    val_loss = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        print('val')
        for  x_temporal, y in val_loader:
            y=y.to(device)
            outputs = model(x_temporal)
            loss = criterion(outputs.squeeze(), y)
            val_loss += loss.item()
            predictions.extend(outputs.squeeze().cpu().numpy())
            true_labels.extend(y.cpu().numpy())
    predictions = np.array(predictions)
    predicted_labels = (predictions > thrs).astype(int)
    y_test=np.array(true_labels)

    
    accuracy = accuracy_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    f1 = f1_score(y_test, predicted_labels)
    auc = roc_auc_score(y_test, predictions)

    print(f'Accuracy: {accuracy:.4f}',f'Precision: {precision:.4f}',f'Recall: {recall:.4f}',f'F1 Score: {f1:.4f}',f'AUC: {auc:.4f}')
    # print(f'Precision: {precision:.4f}')
    # print(f'Recall: {recall:.4f}')
    # print(f'F1 Score: {f1:.4f}')
    # print(f'AUC: {auc:.4f}')
    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')
    if auc>best_auc:
        best_model_wts = model.state_dict()
        best_auc=auc
print("Training Complete")
print('best_auc:',best_auc)
# load
model.load_state_dict(best_model_wts)
model.eval()
val_loss = 0
predictions = []
true_labels = []
with torch.no_grad():
    print('val')
    for  x_temporal, y in test_loader:
        y=y.to(device)
        outputs = model( x_temporal)
        loss = criterion(outputs.squeeze(), y)
        val_loss += loss.item()
        predictions.extend(outputs.squeeze().cpu().numpy())
        true_labels.extend(y.cpu().numpy())
predictions = np.array(predictions)
np.save(f'predictions_full_{i}.npy',predictions)
predicted_labels = (predictions > thrs).astype(int)
y_test=np.array(true_labels)

# metric
accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)
auc = roc_auc_score(y_test, predictions)

print(f'Accuracy: {accuracy:.4f}',f'Precision: {precision:.4f}',f'Recall: {recall:.4f}',f'F1 Score: {f1:.4f}',f'AUC: {auc:.4f}')
predictions = []
true_labels = []
val_loss = 0
with torch.no_grad():
    print('train_val')
    for  x_temporal, y in train_loader:
        y=y.to(device)
        outputs = model( x_temporal)
        loss = criterion(outputs.squeeze(), y)
        val_loss += loss.item()
        predictions.extend(outputs.squeeze().cpu().numpy())
        true_labels.extend(y.cpu().numpy())
predictions = np.array(predictions)
np.save(f'train_predictions_full_{i}.npy',predictions)
predicted_labels = (predictions > thrs).astype(int)
y_test=np.array(true_labels)
#np.save('y_test.npy',y_test)
# metric
accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)
auc = roc_auc_score(y_test, predictions)

print(f'Accuracy: {accuracy:.4f}',f'Precision: {precision:.4f}',f'Recall: {recall:.4f}',f'F1 Score: {f1:.4f}',f'AUC: {auc:.4f}')
