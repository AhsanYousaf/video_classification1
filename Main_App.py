import os
from tkinter import *
from tkinter import messagebox, filedialog
from os import system
import cv2
import moviepy.editor
import pymysql
import warnings
warnings.filterwarnings('ignore')
import tkvideo as tkv
root=Tk()
#upload= Tk()
import os
import tensorflow as tf
import cv2
import numpy as np

from tqdm import tqdm


model_path='models'
# load json and create model
json_file = open(os.path.join(model_path,'model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(os.path.join(model_path,'model.h5'))
print("Loaded model from disk")

model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

classes={0:'basketball',1: 'boxing', 
         2:'cricket',3: 'formula1', 
         4:'kabaddi', 5:'swimming', 
         6:'table_tennis',7: 'weight_lifting'}

for sport_folder in classes:
    folder_path=os.path.join('data',classes[sport_folder])
    if not os.path.exists(folder_path):
        print('Making Path: ',folder_path)
        os.makedirs(folder_path)



def spliting(filename):
    cam = cv2.VideoCapture(filename)
    total_fps=int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    videoename = os.path.basename(filename)

    try:

        # creating a folder named data
        if not os.path.exists('data/'+videoename.rsplit('.', 1)[0]):
            os.makedirs('data/'+videoename.rsplit('.', 1)[0])

        # if not created then raise error
    except OSError:
        print('Error: Creating directory ')
    #audio extraction
    video = moviepy.editor.VideoFileClip(filename)  # Entering the videofile
    audio = video.audio
    audio.write_audiofile(r"./data/"+ videoename.rsplit('.', 1)[0]+"/"+videoename.rsplit('.', 1)[0]+".mp3")

    prediction_counts={'basketball': 0, 'boxing': 0, 
                       'cricket': 0, 'formula1': 0, 'kabaddi': 0, 
                       'swimming': 0, 'table_tennis': 0, 'weight_lifting': 0}
    # frame
    currentframe = 0
    print('Predicting video...')
    pbar = tqdm(total=total_fps)
    while (True):
        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating

            # name = './data/'+ videoename.rsplit('.', 1)[0]+"/F" + str(currentframe) + '.jpg'
            # print('Creating...' + name)
            pred_img = cv2.resize(frame,(224,224))
            pred_img=np.expand_dims(pred_img, axis=0)
            # print(pred_img.shape)
            prediction = model.predict(pred_img)
            maxindex = int(np.argmax(prediction))
            sport=classes[maxindex]
            prediction_counts[sport]=prediction_counts[sport]+1
            pbar.update(1)
            
            currentframe += 1
        else:
            cam.release()
            cv2.destroyAllWindows()
            #currentframe-=1  
            break
    
    max_pred_sport = max(prediction_counts, key=prediction_counts.get)
    
    pred_=(prediction_counts[max_pred_sport]/total_fps)*100
    
    print('Prediction percentage: '+str(pred_)+'%')
    
    vid_path="data/"+max_pred_sport+'/'+max_pred_sport+'_'+videoename.rsplit('.', 1)[0]+'.avi'
  
    print("Writing Video in: ",vid_path)
    cam = cv2.VideoCapture(filename)
    fps = cam.get(cv2.CAP_PROP_FPS)
    fourcc = 'mp4v'
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*fourcc),fps,(w, h))
    pbar = tqdm(total=total_fps)
    while(cam.isOpened()):
          ret, frame = cam.read()
          if ret == True:
              image = cv2.putText(frame, max_pred_sport, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
              vid_writer.write(image)
              pbar.update(1)
          else:
                print('Video writing process complete')
                cam.release()
                cv2.destroyAllWindows()
                #currentframe-=1  
                break
    # cam.release()
    # cv2.destroyAllWindows()
    return currentframe


class Main_App:

    def __init__(self, root):
        self.root = root
        self.root.geometry("500x500")
        self.root.resizable(False,False)
        self.root.configure(background="black")
        self.root.title("Main_App")
        btn_bar=Frame(root,width=500,highlightbackground='white',height='40').place(x='0',y='5')
        Frame.txt_name = Entry(root)
        Frame.txt_name.place(x=300, y=12)
        # print(Frame.txt_name)
        btn_search = Button(root, text='search', width=8, bg='black', fg='white', command=self.search).place(x=430, y=9)
        btn_upload =Button(root, text='upload', width=8, bg='black', fg='white', command=self.upload).place(x=230, y=9)
        btn_logout =Button(root, text='Logout', width=8, bg='black', fg='white', command=self.logout).place(x=5, y=9)
        video_box = Frame(root, width=400, highlightbackground='white', height='300').place(x='50', y='100')
        my_label = Label(root)
        my_label.place(x='50', y='100')
        player = tkv.tkvideo("test/example1.mp4",my_label, loop=1 , size=(400,300))
        player.play()
        btn_next = Button(root, text='>', width=4, bg='black', fg='white', command=self.next).place(x=250, y=450)
        btn_back = Button(root, text='<', width=4, bg='black', fg='white', command=self.back).place(x=210, y=450)
#to search the user required category
    def search(self):
        messagebox.showinfo("search", "we are searching here", parent=self.root)


    def upload(self):
        filename = filedialog.askopenfilename(initialdir="/", title="select a file",
                                              filetype=(("mp4", "*.mp4"), ("All Files", "*.*")))
        if(filename!=""):
            frameno=spliting(filename)
            no_f =str(frameno)
            videoename = os.path.basename(filename)
            try:
                connection = pymysql.connect(host="localhost", user="root", password="", database="db_connectivity")
                cursor = connection.cursor()
                cursor.execute("insert into video_db (video) values (%s)",(filename))
                connection.commit()
                connection.close()
                messagebox.showinfo("Success", "Successfuly upload\n video splite into "+no_f+" frames and audio saved in /data/ "+ videoename.rsplit('.', 1)[0], parent=self.root)
                # system('Main_App.py')
            except Exception as es:
                messagebox.showerror("Error", f"Error due to:{str(es)}", parent=self.root)


    def logout(self):
        root.destroy()
        system('User_Login.py')
    def next(self):
        messagebox.showinfo("next", "move to next video", parent=self.root)
    def back(self):
        messagebox.showinfo("back", "go to previous video", parent=self.root)
obj = Main_App(root)
root.mainloop()
