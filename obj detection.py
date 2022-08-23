#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 # pip install opencv-python


# In[2]:


import matplotlib.pyplot as plt # pip install matplotlib


# In[3]:


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'


# In[4]:


model = cv2.dnn_DetectionModel(frozen_model,config_file)


# In[5]:


classLabels = [] ## empty list of python
file_name = 'Labels.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    #classLabels.append(fpt.read())


# In[6]:


print(classLabels)


# In[7]:


#read an image


# In[8]:


model.setInputSize(320,320)
model.setInputScale(1.0/127.5) ## 255/2 =127.5
model.setInputMean((127.5,127.5,127.5)) ##
model.setInputSwapRB(True)


# In[9]:


img = cv2.imread('caf423cd724487eeb6ac0254cdea36f0.jpg')


# In[10]:


plt.imshow(img)


# In[11]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[12]:


ClassIndex, confidence, bbox = model.detect(img,confThreshold = 0.5)


# In[13]:


print(ClassIndex)


# In[14]:


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img,boxes,(255, 0, 0), 2 )
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)


# In[15]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[16]:


cap = cv2.VideoCapture("videoplayback.mp4")

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("cannot open video")
        
        
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret,frame = cap.read()
    
    classIndex, confidence, bbox = model.detect(frame,confThreshold = 0.55)
    
    print(ClassIndex)
    if (len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence, bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame,boxes,(255, 0, 0), 2 )
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale= font_scale, color = (0, 255, 0), thickness=3 )
                
                
    cv2.imshow("Detection", frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




