#path to YOLO directory
MODEL_PATH = "yolo"
# initialize minimum probability to filter weak detections
MIN_CONF = 0.3
NMS_THRESH = 0.3
#config
#count the total number of people(true/false)
People_Counter = True
#threading (true/false)
Thread = False
#setting the threshold value for total violation limit
Threshold = 15
#ip camera url (0 for real time camera)
url = ''
#if you want an email alert write true
ALERT = False
#write your email (xxx@xxx.xx)
MAIL = ''
#using gpu ( true  false)
USE_GPU = True
#max and min distance
MAX_DISTANCE = 80
MIN_DISTANCE = 50

