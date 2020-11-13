from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
import os
from pprint import pprint 

# if you want to change model, there are few steps you need to follow
# step1: model name
# step2: lable list
# step3: testing data location
# step4: images name in testing folder

# load the training model
net = load_model('labeling-model-1020.h5')

cls_list = ['DB', 'DL','LB','OL','QB','RB','TE','WR']
pred_label = [0] * len(cls_list)
confusion_table = []

for label in cls_list:

    # testing files location
    os.chdir(f'C:/Users/jkxwo/Documents/FootBall Research/02_SecondNetWork/ResNet50_Labeling/dataset_combine/test/{label}') # actual path

    files = []
    for f in os.listdir():
        file_name, file_ext = (os.path.splitext(f))
        if file_ext == '.png':
            files.append(f)


    # predict every image
    for f in files:
        img = image.load_img(f, target_size=(224, 224))
        if img is None:
            continue
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        pred = net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        
        print('image:',f) 
        file_name, file_ext = (os.path.splitext(f))
        label, _ = file_name.split('-') 
        print('Actual Label:', label) # here I can get actual label
        print('Prediction Lable:', cls_list[top_inds[0]]) # here I can get the predict label
        
        # counting prediction label for confusion tabel
        for i in range(len(cls_list)):
            if cls_list[top_inds[0]] == cls_list[i]:
                pred_label[i] += 1


        # print top 5 prediction
        for i in top_inds:
            print('    {:.3f}  {}'.format(pred[i], cls_list[i]))

    # In 'OL' predition, if prediction is QB put into 'OL'(as a Center)  
    if label == 'OL':
        pred_label[3] += pred_label[4]
        pred_label[4] = 0

    confusion_table.append(pred_label)
    print(f'class_list: {cls_list}')
    print(f'predict_label: {pred_label}')
    pred_label = [0] * len(cls_list)


# sum every single line for total autual number and recall
for i in range(len(confusion_table)):
    total_sum = sum(confusion_table[i])
    recall = round(confusion_table[i][i] / total_sum * 100,2) / 100
    confusion_table[i].append(total_sum)
    confusion_table[i].append(recall)


# put in array in order to transpose
confusion_table = np.array(confusion_table)

# transpost the table
trans_ps_confu_table = np.array(confusion_table).T

# turn bake to list, in order to calculate total predicted number and precision 
trans_ps_confu_table = trans_ps_confu_table.tolist()

for i in range(len(trans_ps_confu_table)-1):
    # total predicted number
    total_sum = sum(trans_ps_confu_table[i])
    trans_ps_confu_table[i].append(total_sum)
    if i != len(trans_ps_confu_table)-2:
        # precision
        precision = round(trans_ps_confu_table[i][i]/total_sum * 100, 2) / 100
        trans_ps_confu_table[i].append(precision)


# adding label recall label and precision label
# adding row
cls_list.append('Total')
cls_list.append('Precision')
trans_ps_confu_table.insert(0, cls_list)

# adding column
for i in range(len(cls_list)+1):
    if i == 0:
        trans_ps_confu_table[i].insert(0,'')
    elif i == len(cls_list)-1:
        trans_ps_confu_table[i].insert(0,'Recall')
    else:   
        trans_ps_confu_table[i].insert(0,cls_list[i])

# ============================================================================= #

# Save the Confusion Table in ResNet50_Labeling folder
import csv


# direct back to saving location folder: ResNet50_Labeling folder
os.chdir(f'C:/Users/jkxwo/Documents/FootBall Research/02_SecondNetWork/ResNet50_Labeling') 

with open('Confusion_Table.csv', 'w', newline='') as file:
    print('Writing Confusion Table ...')
    a = csv.writer(file, delimiter=',')
    a.writerows(trans_ps_confu_table)
    print('Finished the writing ...')

