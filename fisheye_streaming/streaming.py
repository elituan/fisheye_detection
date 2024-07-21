import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, torch
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, LazyConfig
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

# Directory where images will be saved
# modelPath =  'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
# modelPath =  'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
# modelPath =  'new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py'
# modelPath =  'Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml'
# modelPath =  'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml' # good here but too long
# modelPath =  'Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml' # best here
modelPath =  'Cityscapes/mask_rcnn_R_50_FPN.yaml' 
threshold = 0.1
ImgScaleFactor= 3
save_dir = './fisheye/imgs/{}_{}'.format(modelPath, threshold)
os.makedirs(save_dir, exist_ok=True)


# Initialize the model
def setup_cfg(modelPath):

    if '.py' not in modelPath:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(modelPath))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(modelPath)
    else:
        # cfg = LazyConfig.load(modelPath)
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg =  LazyConfig.to_py(model_zoo.get_config(modelPath))
        # cfg.merge_from_file(model_zoo.get_config_file(modelPath))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(modelPath)
    
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('https://github.com/facebookresearch/detectron2/blob/main/configs/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py')
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

    return cfg

def detect_vehicles(frame, predictor):
    outputs = predictor(frame)
    # Metadata for the COCO dataset to label detected objects
    v = Visualizer(frame[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2,
                #    instance_mode=ColorMode.IMAGE_BW   # removes the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_frame = v.get_image()[:, :, ::-1]
    return result_frame

def scale_up_img(image, factor):
    height, width = image.shape[:2]

    # Calculate the new dimensions
    new_width = width * factor
    new_height = height * factor

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return


if __name__ == "__main__":
    # video_source  = 'rtsp://guest:UTC_Lot_Monitoring@state-nw-rooftop.cameras.utc.edu:554/axis-media/media.amp'
    # video_source  = 'http://consumer:IyKv4uY7%g^8@10.197.5.92/axis-cgi/mjpg/video.cgi?resolution=1920x1080'
    video_source = ['http://data.utccuip.com/peeples/image.png']

    # Capture video from RTSP
    cap = cv2.VideoCapture(video_source[0])
    cfg = setup_cfg(modelPath)
    predictor = DefaultPredictor(cfg)
    # predictor = model_zoo.get(modelPath)

    img_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        img_counter+=1
        if img_counter >= 3:  # for example, save 10 images
            break

        # Detect vehicles
        # frame = scale_up_img(frame, ImgScaleFactor)

        # processed_frame = detect_vehicles(frame, predictor)
        processed_frame = frame
                
        # Save frame as JPEG file
        img_name = os.path.join(save_dir, f"frame_{img_counter}.jpg")
        cv2.imwrite(img_name, processed_frame)
        print(f"Saved {img_name}")
        
        # Break after saving a specific number of images


        
    cap.release()
    cv2.destroyAllWindows()