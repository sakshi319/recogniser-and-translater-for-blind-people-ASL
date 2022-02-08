import cv2
import time
import numpy as np
import pyttsx3
#from gtts import gTTS
#import os

def nothing(x):
    pass

image_x, image_y = 64,64


from keras.models import load_model
classifier = load_model('Trained_model.h5')
classifier._make_predict_function()

def predictor1():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'All the students'
       elif result[0][1] == 1:
              return 'take out books'
       elif result[0][2] == 1:
              return 'come to respective classroom'
       elif result[0][3] == 1:
              return 'do the given task'
       elif result[0][4] == 1:
              return 'exams'
       elif result[0][5] == 1:
              return 'Fill the respective forms'
       elif result[0][6] == 1:
              return 'Good job'
       elif result[0][7] == 1:
              return 'H'
       elif result[0][8] == 1:
              return 'I'
       elif result[0][9] == 1:
              return 'J'
       elif result[0][10] == 1:
              return 'Keep the belongings here'
       elif result[0][11] == 1:
              return 'lunch time'
       elif result[0][12] == 1:
              return 'Make assignments ready'
       elif result[0][13] == 1:
              return 'No'
       elif result[0][14] == 1:
              return 'Observe the questions or answers'
       elif result[0][15] == 1:
              return 'Pay attention'
       elif result[0][16] == 1:
              return 'any questions'
       elif result[0][17] == 1:
              return 'start Reading'
       elif result[0][18] == 1:
              return 'Stand up'
       elif result[0][19] == 1:
              return 'Time starts now'
       elif result[0][20] == 1:
              return 'Use mobile'
       elif result[0][21] == 1:
              return 'Very good'
       elif result[0][22] == 1:
              return 'start Writing'
       elif result[0][23] == 1:
              return 'wrong answer'
       elif result[0][24] == 1:
              return 'Yes'
       elif result[0][25] == 1:
              return 'Z'

def predictor2():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'A'
       elif result[0][1] == 1:
              return 'B'
       elif result[0][2] == 1:
              return 'C'
       elif result[0][3] == 1:
              return 'D'
       elif result[0][4] == 1:
              return 'E'
       elif result[0][5] == 1:
              return 'F'
       elif result[0][6] == 1:
              return 'G'
       elif result[0][7] == 1:
              return 'H'
       elif result[0][8] == 1:
              return 'I'
       elif result[0][9] == 1:
              return 'J'
       elif result[0][10] == 1:
              return 'K'
       elif result[0][11] == 1:
              return 'L'
       elif result[0][12] == 1:
              return 'M'
       elif result[0][13] == 1:
              return 'N'
       elif result[0][14] == 1:
              return 'O'
       elif result[0][15] == 1:
              return 'P'
       elif result[0][16] == 1:
              return 'Q'
       elif result[0][17] == 1:
              return 'R'
       elif result[0][18] == 1:
              return 'S'
       elif result[0][19] == 1:
              return 'T'
       elif result[0][20] == 1:
              return 'U'
       elif result[0][21] == 1:
              return 'V'
       elif result[0][22] == 1:
              return 'W'
       elif result[0][23] == 1:
              return 'X'
       elif result[0][24] == 1:
              return 'Y'
       elif result[0][25] == 1:
              return 'Z'

def predictor3():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'All set to fly'
       elif result[0][1] == 1:
              return 'Business class'
       elif result[0][2] == 1:
              return 'Check in'
       elif result[0][3] == 1:
              return 'Decorate'
       elif result[0][4] == 1:
              return 'flight booking'
       elif result[0][5] == 1:
              return 'Food storage'
       elif result[0][6] == 1:
              return 'Gate open'
       elif result[0][7] == 1:
              return 'Air hostess'
       elif result[0][8] == 1:
              return 'I'
       elif result[0][9] == 1:
              return 'Jets'
       elif result[0][10] == 1:
              return 'K'
       elif result[0][11] == 1:
              return 'smooth landing'
       elif result[0][12] == 1:
              return 'crew member'
       elif result[0][13] == 1:
              return 'Normal height fly'
       elif result[0][14] == 1:
              return 'Oxygen mask'
       elif result[0][15] == 1:
              return 'Private jet'
       elif result[0][16] == 1:
              return 'Queue'
       elif result[0][17] == 1:
              return 'Reporting area'
       elif result[0][18] == 1:
              return 'Security guards'
       elif result[0][19] == 1:
              return 'Tickets'
       elif result[0][20] == 1:
              return 'Users passport'
       elif result[0][21] == 1:
              return 'Vip class'
       elif result[0][22] == 1:
              return 'Weight limit is 15 kg'
       elif result[0][23] == 1:
              return 'take off anouncemnet'
       elif result[0][24] == 1:
              return 'Y'
       elif result[0][25] == 1:
              return 'pilot zone'

def predictor4():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'Attend meetings'
       elif result[0][1] == 1:
              return 'be in time'
       elif result[0][2] == 1:
              return 'Confetrence call'
       elif result[0][3] == 1:
              return 'Documents'
       elif result[0][4] == 1:
              return 'E'
       elif result[0][5] == 1:
              return 'get files ready'
       elif result[0][6] == 1:
              return 'gather in project room'
       elif result[0][7] == 1:
              return 'H'
       elif result[0][8] == 1:
              return 'Information collection'
       elif result[0][9] == 1:
              return 'J'
       elif result[0][10] == 1:
              return 'Keep confidential'
       elif result[0][11] == 1:
              return 'Lunch'
       elif result[0][12] == 1:
              return 'Meetings'
       elif result[0][13] == 1:
              return 'No excuse'
       elif result[0][14] == 1:
              return 'O'
       elif result[0][15] == 1:
              return 'Print the reoprt'
       elif result[0][16] == 1:
              return 'be Quick'
       elif result[0][17] == 1:
              return 'Reports ready'
       elif result[0][18] == 1:
              return 'Set schedule for team'
       elif result[0][19] == 1:
              return 'do Task'
       elif result[0][20] == 1:
              return 'User requirements'
       elif result[0][21] == 1:
              return 'Video conference'
       elif result[0][22] == 1:
              return 'Webinar'
       elif result[0][23] == 1:
              return 'Xerox machine'
       elif result[0][24] == 1:
              return 'Y'
       elif result[0][25] == 1:
              return 'Z'

def predictor5():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'food item available'
       elif result[0][1] == 1:
              return 'Buffet'
       elif result[0][2] == 1:
              return 'Customise order'
       elif result[0][3] == 1:
              return 'Drinks'
       elif result[0][4] == 1:
              return 'Eggless food'
       elif result[0][5] == 1:
              return 'Fast food menu'
       elif result[0][6] == 1:
              return 'Garden sitting'
       elif result[0][7] == 1:
              return 'any help'
       elif result[0][8] == 1:
              return 'Ice cream section'
       elif result[0][9] == 1:
              return 'Jjunk section'
       elif result[0][10] == 1:
              return 'Ketters'
       elif result[0][11] == 1:
              return 'Lunch options'
       elif result[0][12] == 1:
              return 'Menu card'
       elif result[0][13] == 1:
              return 'food for number of people'
       elif result[0][14] == 1:
              return 'Order'
       elif result[0][15] == 1:
              return 'Place order'
       elif result[0][16] == 1:
              return 'Q'
       elif result[0][17] == 1:
              return 'order is Ready'
       elif result[0][18] == 1:
              return 'Self service here'
       elif result[0][19] == 1:
              return 'Tables available'
       elif result[0][20] == 1:
              return 'U'
       elif result[0][21] == 1:
              return 'Veg food'
       elif result[0][22] == 1:
              return 'Which dish u want'
       elif result[0][23] == 1:
              return 'X'
       elif result[0][24] == 1:
              return 'Yes'
       elif result[0][25] == 1:
              return 'bar zone'

def get_frame(scen):
    camera_port=0
    camera=cv2.VideoCapture(camera_port) #this makes a web cam object
    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    #cv2.namedWindow("test")

    img_counter = 0

    img_text = ''
    #time.sleep(2)

    while True:
        #scen = session.get('scen',None)
        print(scen)
        ret, img = camera.read()
        img = cv2.flip(img,1)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")


        img = cv2.rectangle(img, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])
        imcrop = img[102:298, 427:623]
        hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        cv2.putText(img, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0))
        #cv2.imshow("test", frame)
        cv2.imshow("mask", mask)
    
        #if cv2.waitKey(1) == ord('c'):
        
        img_name = "1.png"
        save_img = cv2.resize(mask, (image_x, image_y))
        cv2.imwrite(img_name, save_img)
        print("{} written!".format(img_name))

        if(scen == 'College'):
              img_text = predictor1()
        elif(scen == 'Home'):
              img_text = predictor2()
        elif(scen == 'Airport'):
              img_text = predictor3()
        elif(scen == 'Office'):
              img_text = predictor4()
        elif(scen == 'Restaurant'):
              img_text = predictor5()

        '''      
        if(img_text):
            engine = pyttsx3.init()
            engine.say(img_text)
            engine.runAndWait()
        '''
        #myobj = gTTS(text=img_text, lang='en', slow=False)
  
        # Saving the converted audio in a mp3 file named 
        # welcome  
        #myobj.save("welcome.mp3") 
  
        # Playing the converted file 
        #os.system("mpg321 welcome.mp3")     
        
        if cv2.waitKey(1) == 27:
            break
        
        imgencode=cv2.imencode('.jpg',img)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

    camera.release()
    cv2.destroyAllWindows()
    del(camera)
