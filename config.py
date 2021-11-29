# encoding: utf-8
class Config:
    detect_model_path_tpu = './model_data/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'
    detect_model_path = './model_data/ssd_mobilenet_v2_face_quant_postprocess.tflite'
    encoding_model_path_tpu = './model_data/facenet_mobilenet_160_quant_edgetpu.tflite'
    encoding_model_path = './model_data/facenet_mobilenet_160_quant.tflite'
    face_database_path = './face_dataset/face_database.db'
    detect_face_threshold = 0.8
    video_source = 'rtsp://admin:12345@10.66.211.212:8554/live'


cfg = Config()
