import yaml
import os
from ultralytics import YOLO

class YOLOSurrogate:
    def __init__(self, config_path="configs/base_config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.model_name = self.cfg['models']['surrogate']['name']
        self.epochs = self.cfg['models']['surrogate']['epochs']
        self.img_size = self.cfg['data']['img_size']
        self.batch_size = self.cfg['data']['batch_size']
        
        # Khởi tạo model (Tự động tải pretrained weights nếu có mạng, hoặc dùng weights local)
        self.model = YOLO(self.model_name) 

    def train(self, data_yaml_path):
        print(f"Bắt đầu train Surrogate Model ({self.model_name})...")
        # device='0' nếu có GPU, 'cpu' nếu không. Ultralytics tự nhận diện.
        self.model.train(
            data=data_yaml_path,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            project="runs/detect",
            name="surrogate_train",
            exist_ok=True,
            verbose=True
        )
        print("Train hoàn tất!")

    def predict(self, image_path):
        # Trả về kết quả để đưa vào Evaluator
        results = self.model.predict(source=image_path, conf=0.25, save=False)
        return results

if __name__ == "__main__":
    # Test luồng đọc config
    wrapper = YOLOSurrogate()
    print(f"Đã load config. Chuẩn bị train model: {wrapper.model_name} trong {wrapper.epochs} epochs.")
    # wrapper.train("dummy_data/DOTA_dummy/data.yaml") # Uncomment để chạy train thử trên dummy data