import os
import sys
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
sys.path.append('yolov7') # Thêm dòng này để fix lỗi ModuleNotFoundError: No module named 'models'
from node import Node, Server, Client

# Đường dẫn để nạp model CNN1D từ thư mục FL của bạn
sys.path.append(r'C:\FederatedLearning\FL\core\model')
from model import CNN1D

class ServerCNN1D(Server):
    def initialize_model(self, weights: str = None) -> None:
        """Khởi tạo mô hình CNN1D từ đầu hoặc từ file checkpoint."""
        model = CNN1D(input_dim=32, num_classes=34) # Bạn có thể điều chỉnh input_dim/num_classes theo data
        if weights and os.path.exists(weights):
            ckpt = torch.load(weights, map_location=self.device)
            if 'model' not in ckpt:
                self._ckpt = {'model': ckpt, 'epoch': -1}
            else:
                self._ckpt = ckpt
            print(f'Server loaded weights from {weights}')
        else:
            self._ckpt = {'model': model, 'epoch': -1}

    def test(self, kround: int, saving_path: str, data: str, bsz: int, imgsz: int, conf: float, iou: float) -> None:
        """Đánh giá mô hình tại Server bằng tập test data toàn cục."""
        # Lưu checkpoint mỗi round
        checkpoint_dir = os.path.join(saving_path, 'weights')
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self._ckpt, os.path.join(checkpoint_dir, f'checkpoint_round_{kround}.pt'))
        
        model = self._ckpt['model'].to(self.device)
        model.eval()
        test_data_path = os.path.join(data, 'global_test_data.pt')
        data_dict = torch.load(test_data_path, map_location=self.device)
        dataset = TensorDataset(data_dict['x'].float(), data_dict['y'].long())
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds) * 100
        
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        
        print(f'[Round {kround}] Server Test - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%')
        
        metrics = {
            'Round': kround,
            'Loss': avg_loss,
            'Accuracy': accuracy,
            'Micro-Precision': p_micro * 100,
            'Micro-Recall': r_micro * 100,
            'Micro-F1': f1_micro * 100,
            'Macro-Precision': p_macro * 100,
            'Macro-Recall': r_macro * 100,
            'Macro-F1': f1_macro * 100,
            'Weighted-Precision': p_weighted * 100,
            'Weighted-Recall': r_weighted * 100,
            'Weighted-F1': f1_weighted * 100
        }
        
        csv_path = os.path.join(saving_path, 'run', 'metrics.csv')
        df = pd.DataFrame([metrics])
        if not os.path.isfile(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)
            
        model.cpu()

    def get_weights(self, metadata: bool):
        """Ghi đè để bỏ ép kiểu sang half-precision (bị lỗi với CNN1D)."""
        weights = copy.deepcopy(self._ckpt) if metadata else copy.deepcopy(self._ckpt['model']).state_dict()
        encrypted_weights, tag, nonce = self._symmetric_encryption(weights)
        encrypted_data = []
        for client_rank in self.clients_public_keys.keys():
            encrypted_data.append((encrypted_weights, tag, nonce))
        return encrypted_data

    def reparameterize(self, architecture: str = 'cnn1d') -> None:
        """Không làm gì vì cnn1d không cần reparameterize như YOLO."""
        self._ckpt_reparam = copy.deepcopy(self._ckpt)

class ClientCNN1D(Client):
    def train(self, nrounds: int, kround: int, epochs: int, architecture: str, data: str, bsz_train: int, imgsz: int,
              cfg: str, hyp: str, workers: int, saving_path: str) -> None:
        """Huấn luyện mô hình ngay trong PyTorch thay vì gọi os.system() của YOLOv7."""
        # Tải dữ liệu iov dựa vào rank của client (rank - 1 vì server rank 0)
        data_path = os.path.join(data, 'federated_data', f'client_{self.rank-1}.pt')
        data_dict = torch.load(data_path, map_location=self.device)
        dataset = TensorDataset(data_dict['x'].float(), data_dict['y'].long())
        self.nsamples = len(dataset)
        dataloader = DataLoader(dataset, batch_size=bsz_train, shuffle=True)
        
        model = self._ckpt['model'].to(self.device)
        w_t = copy.deepcopy(model.state_dict())
        
        model.train()
        # Cấu hình optimizer (có thể chuyển thành args tuỳ ý)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        # Tính delta (cập nhật update gửi về Server)
        w_it = model.state_dict()
        delta_it = {}
        for key in w_t.keys():
            delta_it[key] = w_t[key] - w_it[key]
            
        self._Client__update = delta_it # Update biến private __update của class Client cha
        self._ckpt['model'] = model.cpu()

    def post_init_update(self, data: str, cfg: str, hyp: str, imgsz: int) -> None:
        pass # Không dùng cho cnn1d
