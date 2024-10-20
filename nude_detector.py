import os
import math
import tempfile
import cv2
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
import time

class NudeDetector:
    LABELS = [
        "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
        "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED",
        "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
    ]

    def __init__(self, model_path, providers=None):
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=C.get_available_providers() if not providers else providers
        )
        model_inputs = self.onnx_session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        self.input_name = model_inputs[0].name
        self.blur_exceptions = self._load_default_exceptions()

    def _load_default_exceptions(self):
        return {label: True for label in self.LABELS}

    def _read_frame(self, frame, target_size=320):
        img_height, img_width = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        aspect = img_width / img_height
        
        if img_height > img_width:
            new_height = target_size
            new_width = int(round(target_size * aspect))
        else:
            new_width = target_size
            new_height = int(round(target_size / aspect))
            
        resize_factor = math.sqrt((img_width**2 + img_height**2) / 
                                (new_width**2 + new_height**2))
        
        img = cv2.resize(img, (new_width, new_height))
        
        pad_x = target_size - new_width
        pad_y = target_size - new_height
        
        pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
        pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]
        
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        img = cv2.resize(img, (target_size, target_size))
        image_data = img.astype("float32") / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0)
        
        return image_data, resize_factor, pad_left, pad_top

    def _postprocess(self, output, resize_factor, pad_left, pad_top):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            
            if max_score >= 0.2:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0:4]
                left = int(round((x - w * 0.5 - pad_left) * resize_factor))
                top = int(round((y - h * 0.5 - pad_top) * resize_factor))
                width = int(round(w * resize_factor))
                height = int(round(h * resize_factor))
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
        detections = []
        
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            detections.append({
                "class": self.LABELS[class_id],
                "score": float(score),
                "box": box
            })
        
        return detections

    def detect_frame(self, frame):
        preprocessed_image, resize_factor, pad_left, pad_top = self._read_frame(
            frame, self.input_width
        )
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        return self._postprocess(outputs, resize_factor, pad_left, pad_top)


class VideoProcessor:
    def __init__(self, detector, blur_threshold=50):
        self.detector = detector
        self.blur_threshold = blur_threshold

    def _should_blur(self, detections):
        exposed_count = sum(1 for d in detections if "EXPOSED" in d["class"])
        return exposed_count > 0

    def _apply_blur(self, frame, detections):
        for detection in detections:
            if "EXPOSED" in detection["class"]:
                box = detection["box"]
                x, y, w, h = box
                if (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1] and 
                    0 <= y + h < frame.shape[0] and 0 <= x + w < frame.shape[1]):
                    region = frame[y:y+h, x:x+w]
                    frame[y:y+h, x:x+w] = cv2.GaussianBlur(region, (99, 99), 30)
        return frame

    def process_video(self, input_video, progress_callback=None):
        cap = None
        out = None
        temp_output = None
        
        try:
            # Create temporary file
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_output_path = temp_output.name
            temp_output.close()  # Close the file handle immediately

            # Open video capture
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                temp_output_path, fourcc, fps, (frame_width, frame_height)
            )

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                detections = self.detector.detect_frame(frame)
                if self._should_blur(detections):
                    frame = self._apply_blur(frame, detections)

                out.write(frame)
                frame_count += 1

                if progress_callback:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress)

            # Properly release resources
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()

            # Small delay to ensure all files are properly closed
            time.sleep(0.1)

            # Read the processed video
            with open(temp_output_path, 'rb') as f:
                processed_video = f.read()

            return processed_video

        except Exception as e:
            raise e

        finally:
            # Clean up resources
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            
            # Small delay before attempting to delete
            time.sleep(0.1)
            
            # Try to remove temporary file
            try:
                if temp_output is not None and os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_output_path}: {e}")