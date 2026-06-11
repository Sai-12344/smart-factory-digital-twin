import numpy as np

class VisionEncoder:
    def encode(self, pil_image):
        return np.zeros(256, dtype=np.float32)

class TextEncoder:
    def encode(self, caption):
        return np.zeros(256, dtype=np.float32)

class FusionModule:
    def encode(self, vision_vec, text_vec):
        return np.concatenate([vision_vec, text_vec], axis=0)

class VLAAgent:
    def __init__(self):
        self.vision = VisionEncoder()
        self.text = TextEncoder()
        self.fusion = FusionModule()
