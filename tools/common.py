import numpy as np

class BBox:
    def __init__(self, x, y, r, b, landmark):
        
        self.x = x
        self.y = y
        self.r = r
        self.b = b
        self.landmark = landmark

    def __repr__(self):
        landmark_info = "HasLandmark" if self.landmark else "NoLandmark"
        return f"{{Face {self.x}, {self.y}, {self.r}, {self.b}, {landmark_info} }}"
    
    @property
    def left_top_i(self):
        return int(self.x), int(self.y)
    
    @property
    def right_bottom_i(self):
        return int(self.r), int(self.b)
    
    @property
    def center_i(self):
        return int((self.x + self.r) * 0.5), int((self.y + self.b) * 0.5)
    
    @property
    def center(self):
        return (self.x + self.r) * 0.5, (self.y + self.b) * 0.5
    
    @property
    def width(self):
        return self.r - self.x + 1
    
    @property
    def height(self):
        return self.b - self.y + 1

    @property
    def location(self):
        return self.x, self.y, self.r, self.b
    
        
class ImageObject:
    def __init__(self, file):
        self.file = file
        self.bboxes = []

    def add(self, annotation):
        x, y, w, h = annotation[:4]
        r = x + w - 1
        b = y + h - 1
        landmark = None
        
        if len(annotation) == 20:
            # x, y, w, h, xyz, xyz, xyz, xyz, xyz, unknow
            landmark = []
            for i in range(5):
                px = annotation[i * 3 + 0 + 4]
                py = annotation[i * 3 + 1 + 4]
                pz = annotation[i * 3 + 2 + 4]
                
                if pz == -1:
                    landmark = None
                    break
                    
                landmark.append([px, py])
        self.bboxes.append(BBox(x, y, r, b, landmark))
        

def load_widerface_annotation(ann_file):
    with open(ann_file, "r") as f:
        lines = f.readlines()

    imageObject = None
    file = None
    images = []
    for line in lines:
        line = line.replace("\n", "")

        if line[0] == "#":
            file = line[2:]
            imageObject = ImageObject(file)
            images.append(imageObject)
        else:
            imageObject.add([float(item) for item in line.split(" ")])
    return images


def draw_gauss(heatmap, x, y, box_size):

    if not isinstance(box_size, tuple):
        box_size = (box_size, box_size)

    box_width, box_height = box_size
    diameter = min(box_width, box_height)

    height, width = heatmap.shape[:2]
    sigma = diameter / 6
    radius = max(1, int(diameter * 0.5))
    s = 2 * sigma * sigma
    ky, kx = np.ogrid[-radius:+radius+1, -radius:+radius+1]
    kernel = np.exp(-(kx * kx + ky * ky) / s)
    
    dleft, dtop = -min(x, radius), -min(y, radius)
    dright, dbottom = +min(width - x, radius+1), +min(height - y, radius+1)
    select_heatmap = heatmap[y+dtop:y+dbottom, x+dleft:x+dright]
    select_kernel = kernel[radius+dtop:radius+dbottom, radius+dleft:radius+dright]
    if min(select_heatmap.shape) > 0:
        np.maximum(select_heatmap, select_kernel, out=select_heatmap)
    return heatmap