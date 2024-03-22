from ultralytics import YOLO
from PIL import Image

# Load a model
#model = YOLO('yolov8n.pt')  # load an official model

#model = YOLO("g:/Pyprojects/Wildfire-Detection-App-main/project_fire_data’/experiment_fire_data/weights/best.pt") # load trained model
#model = YOLO("g:/Pyprojects/Wildfire-Detection-App-main/fire-models/fire_m.pt") # load trained model
model = YOLO("g:/Pyprojects/Wildfire-Detection-App-main/project_combineData/combineData_16_epo3/weights/best.pt") # load trained model


# Run batched inference on a list of images
'''
results = model(['g:/Pyprojects/Wildfire-Detection-App-main/dataset/combineData/train/images/1_mp4-5_jpg.rf.b26e281ebe629ca7346870e04254772a.jpg', 'g:/Pyprojects/Wildfire-Detection-App-main/dataset/datamy/test/images\fire.22.png'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
'''
results = model(['https://www.riatomsk.ru/Upload/sub-10/50057_4.jpg'])  # return a list of Results objects
'''
results = model(['g:/Pyprojects/Wildfire-Detection-App-main/src/orig.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/2.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/3.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/4.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/5.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/6.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/7.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/8.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/9.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/11.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/12.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/13.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/14.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/15.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/16.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/17.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/18.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/19.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/20.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/21.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/22.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/23.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/24.jpg',
                 'g:/Pyprojects/Wildfire-Detection-App-main/data_check/25.jpg'])  # return a list of Results objects
'''
# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes


# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    #im.save('results.jpg')  # save image
