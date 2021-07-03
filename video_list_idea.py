# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 23:38:43 2021

@author: Ahsan
"""
import cv2
from difflib import SequenceMatcher
import os
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

similarity_dict={'basketball': 0, 'boxing': 0, 
                   'cricket': 0, 'formula1': 0, 'kabaddi': 0, 
                   'swimming': 0, 'table_tennis': 0, 'weight_lifting': 0}

vide_search=input('ADD:')# taking input


for key in similarity_dict:
    score=similar(vide_search,key)
    similarity_dict[key]=score
        
max_sxore_input = max(similarity_dict, key=similarity_dict.get)
print('Searching in: ',max_sxore_input)

files_list=os.listdir(os.path.join('data',max_sxore_input))
print(files_list)# list of files in searched directory

# source=list_selection


# cap = cv2.VideoCapture(source)
# while(cap.isOpened()):
#   ret, frame = cap.read()
#   if ret == True:
#       # print(frame.shape)
#       cv2.imshow('Requested Video',frame)
#       if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()