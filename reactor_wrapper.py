import sys
import os

# Añadir el directorio de dependencias al sys.path
sys.path.insert(0, "/content/invokeai4.2.8/reactor_deps")

from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import cv2
import numpy as np
from PIL import Image
import logging

class ReactorWrapper:
    _instance = None

    def __init__(self):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        MODEL_PATH = "/content/invokeai4.2.8/models/inswapper_128.onnx"
        self.app = FaceAnalysis(name="buffalo_l", root="/content/invokeai4.2.8/models", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

        # Configurar ESRGAN
        model_path = "/content/invokeai4.2.8/models/ESRGAN_8x_NMKD-Superscale_150000_G.pth"
        self.esrgan_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        loadnet = torch.load(model_path, map_location="cpu")
        if "params_ema" in loadnet:
            self.esrgan_model.load_state_dict(loadnet["params_ema"], strict=False)
        else:
            self.esrgan_model.load_state_dict(loadnet, strict=False)
        self.esrgan_model.eval()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ReactorWrapper()
        return cls._instance

    def swap_face(self, input_img, source_img, restore_face=True):
        """Realiza face swap usando insightface y seamlessClone, con opción de restauración ESRGAN."""
        input_rgb = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
        source_rgb = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    
        # Validar dimensiones de las imágenes
        if input_rgb.shape[0] == 0 or input_rgb.shape[1] == 0 or source_rgb.shape[0] == 0 or source_rgb.shape[1] == 0:
            raise ValueError("Las imágenes de entrada o fuente tienen dimensiones inválidas.")
    
        # Detectar rostros
        source_faces = self.app.get(source_rgb)
        input_faces = self.app.get(input_rgb)
        logging.debug(f"Source faces detected: {len(source_faces)}, Input faces detected: {len(input_faces)}")
        if not source_faces or not input_faces:
            raise ValueError("No se detectaron caras en la imagen fuente o de entrada.")
    
        source_face = source_faces[0]
        input_face = input_faces[0]
        logging.debug(f"Source face bbox: {source_face.bbox}, Input face bbox: {input_face.bbox}")
        logging.debug(f"Source face keypoints: {source_face.kps}, Input face keypoints: {input_face.kps}")
    
        # Bounding box del rostro objetivo y fuente
        t_bbox = input_face.bbox.astype(int)
        s_bbox = source_face.bbox.astype(int)
    
        # Validar bounding boxes
        s_bbox = validate_bbox(s_bbox, source_rgb.shape, "source_image")
        t_bbox = validate_bbox(t_bbox, input_rgb.shape, "input_image")
    
        # Extraer y redimensionar el rostro fuente
        source_cropped = source_rgb[s_bbox[1]:s_bbox[3], s_bbox[0]:s_bbox[2]]
        if source_cropped.shape[0] == 0 or source_cropped.shape[1] == 0:
            raise ValueError("La región recortada del rostro fuente tiene dimensiones inválidas.")
    
        h, w = t_bbox[3] - t_bbox[1], t_bbox[2] - t_bbox[0]
        if h <= 0 or w <= 0:
            raise ValueError(f"Dimensiones inválidas para el rostro objetivo: ancho={w}, alto={h}")
    
        source_cropped = cv2.resize(source_cropped, (w, h), interpolation=cv2.INTER_AREA)
    
        # Crear máscara para el rostro objetivo
        mask = np.zeros(input_rgb.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(np.int32(input_face.kps))
        cv2.fillConvexPoly(mask, hull, 255)
        if not np.any(mask):
            raise ValueError("La máscara del rostro objetivo está vacía.")
    
        # Verify mask and source_cropped dimensions
        if mask.shape[:2] != (input_rgb.shape[0], input_rgb.shape[1]):
            raise ValueError(f"La máscara tiene dimensiones incorrectas: {mask.shape[:2]}, esperado: {input_rgb.shape[:2]}")
        if source_cropped.shape[:2] != (h, w):
            raise ValueError(f"Source cropped tiene dimensiones incorrectas: {source_cropped.shape[:2]}, esperado: ({h}, {w})")
    
        # Validar centro para seamlessClone
        center = ((t_bbox[0] + t_bbox[2]) // 2, (t_bbox[1] + t_bbox[3]) // 2)
        if center[0] < 0 or center[1] < 0 or center[0] >= input_rgb.shape[1] or center[1] >= input_rgb.shape[0]:
            logging.error(f"Centro inválido para seamlessClone: {center}, imagen: {input_rgb.shape}")
            raise ValueError(f"Centro inválido para seamlessClone: {center}")
        logging.debug(f"Centro para seamlessClone: {center}")
    
        # Realizar seamless clone
        try:
            result = cv2.seamlessClone(source_cropped, input_rgb, mask, center, cv2.NORMAL_CLONE)
        except cv2.error as e:
            logging.error(f"Error en cv2.seamlessClone: {e}")
            raise ValueError(f"Fallo en seamlessClone: {e}")
    
        # Restauración con ESRGAN
        if restore_face:
            try:
                tensor = torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    output = self.esrgan_model(tensor).clamp_(0, 1)
                output = (output.squeeze().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                result = output
            except Exception as e:
                logging.warning(f"Error en ESRGAN: {e}. Usando imagen sin restaurar.")
                result = result
    
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))        
