import numpy as np

class RobustnessEvaluator:
    """
    Bộ đánh giá độ bền vững, tính toán ASR và Độ ổn định nền (Background Stability).
    """
    def __init__(self, iou_threshold=0.5, conf_threshold=0.25):
        self.iou_thresh = iou_threshold
        self.conf_thresh = conf_threshold

    def _calc_iou(self, box1, box2):
        # box format: [xmin, ymin, xmax, ymax]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0

    def evaluate(self, clean_preds, patched_preds, target_bbox):
        """
        clean_preds: List các box [xmin, ymin, xmax, ymax, conf, cls] của ảnh sạch
        patched_preds: List các box của ảnh đã dán patch
        target_bbox: [xmin, ymin, xmax, ymax] của mục tiêu đang bị tấn công
        """
        # Lọc bỏ các prediction có confidence thấp
        clean_boxes = [b for b in clean_preds if b[4] >= self.conf_thresh]
        patched_boxes = [b for b in patched_preds if b[4] >= self.conf_thresh]
        
        # 1. TÍNH ASR (Attack Success Rate)
        # Mục tiêu biến mất = Không có box nào của patched_preds khớp (IoU > thresh) với target_bbox
        target_detected = False
        for p_box in patched_boxes:
            if self._calc_iou(p_box[:4], target_bbox) >= self.iou_thresh:
                target_detected = True
                break
        asr = 1.0 if not target_detected else 0.0
        
        # 2. TÍNH ĐỘ ỔN ĐỊNH NỀN (Background Stability / False Positives)
        # Đếm số lượng box "mọc thêm" ở ngoài vùng target_bbox so với ảnh sạch
        # Đây là chỉ số L_fp trong hàm Fitness của DE
        def is_outside(box, target):
            # Kiểm tra tâm của box có nằm ngoài target_bbox không
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            return not (target[0] <= cx <= target[2] and target[1] <= cy <= target[3])
            
        clean_fp = len([b for b in clean_boxes if is_outside(b[:4], target_bbox)])
        patched_fp = len([b for b in patched_boxes if is_outside(b[:4], target_bbox)])
        
        # Số lượng báo động giả SINH THÊM do patch gây ra
        new_false_positives = max(0, patched_fp - clean_fp)
        
        # 3. TÍNH MỨC GIẢM OBJECTNESS (AORR - Average Objectness Reduction Rate)
        clean_max_conf = max([b[4] for b in clean_boxes if self._calc_iou(b[:4], target_bbox) >= self.iou_thresh], default=0.0)
        patched_max_conf = max([b[4] for b in patched_boxes if self._calc_iou(b[:4], target_bbox) >= self.iou_thresh], default=0.0)
        
        aorr = clean_max_conf - patched_max_conf
        
        return {
            "ASR": asr,
            "AORR": aorr,
            "New_False_Positives": new_false_positives,
            "Clean_Detections": len(clean_boxes),
            "Patched_Detections": len(patched_boxes)
        }

# --- CHẠY TEST THỬ TRÊN DỮ LIỆU GIẢ LẬP ---
if __name__ == "__main__":
    evaluator = RobustnessEvaluator()
    
    # Giả lập ảnh sạch: Model phát hiện ra máy bay (target) và 1 cái cây (nền)
    target_bbox = [100, 100, 300, 300] 
    clean_preds = [
        [105, 105, 295, 295, 0.95, 0],  # Máy bay (Conf 0.95)
        [500, 500, 600, 600, 0.80, 1]   # Cái cây (Nền)
    ]
    
    # Giả lập ảnh bị dán Patch DE: 
    # Kịch bản 1: Patch TỐT (Máy bay biến mất, không sinh thêm quái vật)
    patched_preds_good = [
        [500, 500, 600, 600, 0.80, 1]   # Chỉ còn cái cây
    ]
    
    # Kịch bản 2: Patch XẤU (Máy bay biến mất, nhưng sinh ra 2 báo động giả ở nền)
    patched_preds_bad = [
        [500, 500, 600, 600, 0.80, 1],  # Cái cây
        [10, 10, 50, 50, 0.60, 1],      # Báo động giả 1
        [800, 800, 900, 900, 0.75, 0]   # Báo động giả 2
    ]
    
    print("--- Kịch bản Patch Tốt ---")
    print(evaluator.evaluate(clean_preds, patched_preds_good, target_bbox))
    # Mong đợi: ASR=1.0, AORR=0.95, New_False_Positives=0
    
    print("\n--- Kịch bản Patch Xấu (Bị bẫy L_fp) ---")
    print(evaluator.evaluate(clean_preds, patched_preds_bad, target_bbox))
    # Mong đợi: ASR=1.0, AORR=0.95, New_False_Positives=2 -> DE sẽ phạt cá thể này!