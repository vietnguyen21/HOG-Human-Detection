import cv2
from utils import run_detection, MODEL

def process_video(video_source=0):
    cap = cv2.VideoCapture(video_source)
    

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    print("[INFO] Starting video stream... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display_size = (320, 240) 
        frame = cv2.resize(frame, display_size)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = run_detection(frame_rgb, step_size=96, threshold=0.5)

        if len(boxes) > 0:
            for (x, y, w, h, score) in boxes:
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"Person: {score:.2f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        cv2.imshow('Human Detection Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(0)