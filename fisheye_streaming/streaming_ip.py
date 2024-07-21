import cv2
import urllib.request
import time
import os
import numpy as np
from datetime import datetime
import pytz


number_frame = 100000 # Number of frame to download 
interval = 5 # Time (second) to wait before download next imgs 

# URL of the image
# urls = ['http://data.utccuip.com/Georgia/image.png','http://data.utccuip.com/market/image.png','http://data.utccuip.com/peeples/image.png','http://data.utccuip.com/lindsay/image.png']
# urls = ['https://admin:jack1234@10.198.0.116:8902/api/camera/00:30:53:27:41:96.png']
streets = ["frazier", "forest", "broad", "market", "georgia", 
           "lindsay", "douglas", "peeples", "magnolia", "central", "chestnut"] #'houston',  get error
urls = ['http://data.utccuip.com/{}/image.png'.format(street) for street in streets]

# save_dir = './fisheye/imgs/15'
gmt_minus_4 = pytz.timezone('Etc/GMT+4')

def fetch_url(url, retries=1000, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            req = urllib.request.urlopen(url)
            return req
        except urllib.error.URLError as e:
            print(f"Attempt {attempt + 1} failed: {e.reason}")
            attempt += 1
            time.sleep(delay)
    
    print("All attempts failed. Continuing with the program.")
    return None


counter = 0
while True:
    # Delay for 1 second
    print('i am working')
    time.sleep(interval)

    # Check the time
    current_time = datetime.now(gmt_minus_4).time()
    hour = current_time.hour
    if hour>=14 and hour<16:
        save_dir = './fisheye/imgs/14'        
    elif hour>=20 and hour<22:
        save_dir = './fisheye/imgs/20'        
    elif hour>=22 and hour<24:
        save_dir = './fisheye/imgs/22'        
    elif hour>=5 and hour<8:
        save_dir = './fisheye/imgs/5'
    else:
        continue

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    counter += 1
    for url in urls:
        print(url)
        # req = urllib.request.urlopen(url)
        try:
            req = urllib.request.urlopen(url)
        except urllib.error.URLError as e:
            # print(f"Attempt {attempt + 1} failed: {e.reason}")
            continue
        
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)

        image = cv2.imdecode(arr, -1)
        # Check if the image is downloaded successfully
        if image is not None:
            # Save the image locally
            street_name = url.split('/')[-2]
            current_time = datetime.now(gmt_minus_4).time()
            img_name = os.path.join(save_dir, f"{street_name}_{counter}_{current_time}.jpg")
            cv2.imwrite(img_name, image)
            print("Saving img {}".format(img_name))
        else:
            print("Error: Image {} could not be downloaded.".format(img_name))


    if counter == number_frame: 
        break