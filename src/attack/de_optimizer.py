import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# --- HÀM GIẢ LẬP (MOCK) THAY CHO MODEL THẬT ---
# Sau này bạn sẽ thay hàm này bằng việc gọi model.predict(img)
def mock_model_inference(patch_pixels):
    """Giả lập Objectness Score và False Positives"""
    # Giả sử patch càng sáng (nhiều pixel 255) thì Objectness càng giảm (tấn công tốt)
    obj_loss = 1.0 - (np.mean(patch_pixels) / 255.0) 
    # Giả lập False Positive (nền bị nhiễu)
    fp_loss = np.std(patch_pixels) / 255.0 
    return obj_loss, fp_loss

# --- CÁC RÀNG BUỘC VẬT LÝ (NPS & TV) ---
def calc_tv_loss(patch_pixels):
    """Total Variation - Ép patch mượt mà, không phải nhiễu tivi"""
    # TV Loss = Tổng độ chênh lệch giữa các pixel liền kề
    diff_x = np.abs(patch_pixels[:, :-1] - patch_pixels[:, 1:])
    diff_y = np.abs(patch_pixels[:-1, :] - patch_pixels[1:, :])
    return np.mean(diff_x) + np.mean(diff_y)

def calc_nps_loss(patch_pixels):
    """Non-Printability Score - Ép màu về bảng màu in được (Giả lập)"""
    # Giả sử chỉ cho phép các mảng màu lớn, phạt các đốm nhỏ lẻ tẻ
    return np.var(patch_pixels) 

# --- HÀM FITNESS TỔNG HỢP (ĐỂ DE TỐI ƯU) ---
def fitness_function(x, *args):
    """
    x: Vector 1D chứa các pixel của patch (đã được flatten)
    """
    patch_size = int(np.sqrt(len(x)))
    patch = x.reshape((patch_size, patch_size))
    
    # 1. Gọi Model (Hiện tại là Mock, sau này thay bằng YOLO/RT-DETR)
    l_obj, l_fp = mock_model_inference(patch)
    
    # 2. Tính ràng buộc vật lý
    l_tv = calc_tv_loss(patch)
    l_nps = calc_nps_loss(patch)
    
    # 3. Trọng số (Lambda) - Có thể chỉnh trong config.yaml
    w1, w2, w3, w4 = 1.0, 0.5, 0.1, 0.1 
    
    total_loss = w1 * l_obj + w2 * l_fp + w3 * l_tv + w4 * l_nps
    return total_loss

# --- VÒNG LẶP TIẾN HÓA VI PHÂN (DE) ---
def run_de_attack(patch_size=32):
    print(f"Bắt đầu tối ưu Patch {patch_size}x{patch_size} bằng DE...")
    # Không gian tìm kiếm: Mỗi pixel có giá trị từ 0 đến 255
    bounds = [(0, 255) for _ in range(patch_size * patch_size)]
    
    # Ghi lại lịch sử loss để vẽ biểu đồ hội tụ (Rất quan trọng để bảo vệ GĐ 4.1)
    history = []
    def callback(xk, convergence):
        current_loss = fitness_function(xk)
        history.append(current_loss)
        
    result = differential_evolution(
        fitness_function, 
        bounds, 
        popsize=10, 
        maxiter=5, # Test nhanh 5 thế hệ
        callback=callback,
        disp=True
    )
    
    # Xuất kết quả
    best_patch = result.x.reshape((patch_size, patch_size)).astype(np.uint8)
    print(f"Tối ưu hoàn tất. Loss thấp nhất: {result.fun:.4f}")
    
    # Vẽ biểu đồ hội tụ (Dùng cho Báo cáo GĐ 4)
    plt.plot(history)
    plt.title("DE Convergence Curve (Fitness vs Generations)")
    plt.xlabel("Generation")
    plt.ylabel("Loss (Lower is better)")
    plt.savefig("de_convergence.png")
    print("Đã lưu biểu đồ hội tụ: de_convergence.png")
    
    return best_patch

if __name__ == "__main__":
    # Chạy test ngay không cần mạng
    run_de_attack(patch_size=16) # Dùng 16x16 để test logic cho nhanh