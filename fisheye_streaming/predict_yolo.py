from ultralytics import YOLO
import os
import cv2 

# model_name = 'yolov9e.pt'
model_name = './fisheye/prediction/weights/yolov9_e_best_checkpoint.pt'

save_dir = './fisheye/prediction/yolov9e'
load_dirs = ['./fisheye/imgs/morning9am/', './fisheye/imgs/morning10am/', './fisheye/imgs/morning11am/', './fisheye/imgs/afternoon5pm/']
# load_dirs = ['./fisheye/imgs/morning9am/']


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_image_paths(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # List of common image file extensions

    image_paths = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(file_path)
    
    return image_paths

def get_save_path(img_path, save_dir):
    img_name = '_'.join(img_path.split('/')[-2:])
    save_path = os.path.join(save_dir,img_name) 
    return save_path

def split_into_chunks(lst, chunk_size):
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


if __name__ == "__main__":
    # Prepare path
    create_dir(save_dir)
    load_img_paths = []
    for dir in load_dirs:
        load_img_paths += get_image_paths(dir)

    # # Build a YOLOv9c model from scratch
    # model = YOLO('yolov9c.yaml')
    # Build a YOLOv9c model from pretrained weight
    model = YOLO(model_name)


    # # Train the model on the COCO8 example dataset for 100 epochs
    # results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

    # Run inference with the YOLOv9c model on the 'bus.jpg' image
    chunk_load_img_paths = split_into_chunks(load_img_paths, 10)
    results = []
    for chunk_load_img_path in chunk_load_img_paths:
        result = model(chunk_load_img_path)
        results += result

    print('predict successful')

    for i,result in enumerate(results):
        save_path = get_save_path(load_img_paths[i], save_dir)
        print('saving img {}'.format(save_path))
        result.save(filename= save_path)

