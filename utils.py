import os
import cv2
import joblib
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
from skimage.feature import hog
from skimage.transform import resize

# Cấu hình hằng số
IMAGE_SIZE = (128, 64)
MODEL_PATH = 'HOG_detection.pkl'

# Load model một lần duy nhất (Singleton pattern)
MODEL = None
if os.path.exists(MODEL_PATH):
    MODEL = joblib.load(MODEL_PATH)
else:
    print(f"Warning: Model file '{MODEL_PATH}' not found. Please ensure it exists.")

def extract_color_histogram(img_patch, bins=(16, 16, 16)):
    """Trích xuất Histogram màu sắc từ ảnh (không gian màu HSV)."""
    hsv = cv2.cvtColor(img_patch, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_features(img):
    """Trích xuất kết hợp HOG (cấu trúc) và Color Histogram (màu sắc)."""
    img_resize = resize(img, IMAGE_SIZE)
    img_resize = img_resize.astype('float32')

    # Chia làm 3 phần theo chiều dọc để lấy thông tin cấu trúc cơ thể
    part1 = img_resize[0:42, :]
    part2 = img_resize[42:84, :]
    part3 = img_resize[84:128, :]
    
    def get_hog(part):
        return hog(part, orientations=9, 
                   pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), 
                   channel_axis=-1, 
                   transform_sqrt=True)

    # Nối các vector đặc trưng
    spatial_fd = np.hstack((get_hog(part1), get_hog(part2), get_hog(part3)))
    fd_color = extract_color_histogram(img_resize)
    
    return np.hstack((spatial_fd, fd_color))

def predict_patch(img_patch):
    """Dự đoán một vùng ảnh có phải là đối tượng hay không."""
    if MODEL is None:
        return None
    
    features = extract_features(img_patch).reshape(1, -1)
    return MODEL.predict(features)[0]

def run_detection(img, step_size=16, threshold=1):
    if MODEL is None:
        return []

    window_size = (64, 128)
    rects = []
    scores = []

    for y in range(0, img.shape[0] - window_size[1], step_size):
        for x in range(0, img.shape[1] - window_size[0], step_size):
            window = img[y:y+window_size[1], x:x+window_size[0]]
            fd = extract_features(window).reshape(1, -1)
            score = MODEL.decision_function(fd)[0]
            
            if score > threshold:
                rects.append([x, y, window_size[0], window_size[1]])
                scores.append(float(score))

    if len(rects) == 0:
        return []

    indices = cv2.dnn.NMSBoxes(rects, scores, score_threshold=threshold, nms_threshold=0.3)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = rects[i]
            final_boxes.append((x, y, w, h, scores[i]))

    return final_boxes

def visualize_results_cv2(img, detected_boxes, window_name='Detection Result'):
    """Thay thế hàm vẽ bằng OpenCV để không bị lỗi Matplotlib."""
    # OpenCV sử dụng BGR, nếu img đầu vào là RGB (từ skimage) thì cần chuyển đổi
    img_display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if len(detected_boxes) > 0:
        for (x, y, w, h, score) in detected_boxes:
            # Vẽ khung chữ nhật
            cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Ghi điểm số
            cv2.putText(img_display, f"Score: {score:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow(window_name, img_display)
    print("Press any key on the image window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Nếu bạn có hàm detection(img) cũ, hãy sửa nó để gọi visualize_results_cv2
def detection(img):
    boxes = run_detection(img)
    visualize_results_cv2(img, boxes)