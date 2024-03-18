from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from flask import Flask, render_template, request, redirect, url_for
import csv
from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from flask import *
from twilio.rest import Client
import pytesseract
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import keras
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense
from keras import optimizers
from keras.applications.resnet50 import ResNet50
import tensorflow as tf
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global classifier
global svm_acc, kmeans_acc
global X, Y
global X_train, X_test, y_train, y_test
global pca

import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename
# Function to perform image processing operations

def process_image(image_path, operation):
    image = cv2.imread(image_path)

    # Perform image processing operations
    if operation == "resize":
        processed_image = cv2.resize(image, (400, 300))
    elif operation == "compress":
        compressed_image = cv2.imwrite("compressed_image.jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 50])
        processed_image = cv2.imread("compressed_image.jpg")
    elif operation == "contour":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        processed_image = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    elif operation == "edge":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.Canny(gray, 100, 200)
    elif operation == "masking":
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(mask, (image.shape[1] // 2, image.shape[0] // 2), 100, 255, -1)
        masked = cv2.bitwise_and(image, image, mask=mask)
        processed_image = masked
    else:
        processed_image = image  # No processing
    #cv2.imshow(processed_image, "processsed image")
    return processed_image


def calculate_ssim(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray_img1, gray_img2, full=True)
    return score


# File upload route
# Calculate SSIM and check if images are forged
@app.route('/check_forgery', methods=['POST'])
def check_forgery():
    if request.method == 'POST':
        # Get uploaded images
        image1 = request.files['image1']
        image2 = request.files['image2']

        # Save uploaded images
        filename1 = secure_filename(image1.filename)
        filename2 = secure_filename(image2.filename)
        image1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        image1.save(image1_path)
        image2.save(image2_path)

        # Load images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        # RESNET-50 Model 
        
        original_cheques= os.listdir('C:\\Users\\prave\\Final_Year_Project\\Dataset\\Training_Data')
        fake_cheques = os.listdir('C:\\Users\\prave\\Final_Year_Project\\Dataset\\Testing_Data')
        # create the labels
        original_cheques_labes = [1]*110

        fake_cheques_labels = [0]*29

        labels = original_cheques_labes + fake_cheques_labels

        # convert images to numpy arrays+

        original_cheque_images_path = 'C:\\Users\\prave\\Final_Year_Project\\Dataset\\Training_Data\\'

        data = []

        for img_file in original_cheques:

            image = Image.open(original_cheque_images_path + img_file)
            image = image.resize((128,128))
            image = image.convert('RGB')
            image = np.array(image)
            data.append(image)

        fake_cheques_path = 'C:\\Users\\prave\\Final_Year_Project\\Dataset\\Testing_Data\\'

        for img_file in fake_cheques:

            image = Image.open(fake_cheques_path + img_file)
            image = image.resize((128,128))
            image = image.convert('RGB')
            image = np.array(image)
            data.append(image)

        # converting image list and label list to numpy arrays

        X = np.array(data)
        Y = np.array(labels)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

        # scaling the data

        X_train_scaled = X_train/255

        X_test_scaled = X_test/255

        X_train_scaled[0]
        from keras.layers import Dropout, Dense
        from keras import optimizers
        from keras.applications.resnet50 import ResNet50
        restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(128,128,3))
        output = restnet.layers[-1].output
        # output = keras.layers.Flatten()(output)

        for layer in restnet.layers:
            layer.trainable = False

        restnet.summary()
        import keras
        model = keras.Sequential()
        model.add(restnet)
        model.add(Dense(512, activation='relu', input_dim=(128,128,3)))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(4, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(lr=2e-5),
        metrics=['accuracy'])
        model.summary()

        num_of_classes = 2
        import keras

        model = tf.keras.Sequential()

        model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


        model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.5))


        model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))

        # compile the neural network
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        model.summary()

        # training the neural network
        history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=15)

        loss, accuracy = model.evaluate(X_test_scaled, Y_test)
        print('Test Accuracy =', accuracy)

        h = history
        # # plot the loss value
        # plt.plot(h.history['loss'], label='train loss')
        # plt.plot(h.history['val_loss'], label='validation loss')
        # plt.legend()
        # plt.show()

        # # plot the accuracy value
        # plt.plot(h.history['accuracy'], label='train accuracy')
        # plt.plot(h.history['val_accuracy'], label='validation accuracy')
        # plt.legend()
        # plt.show()
    
        #input_image_path = input('Path of the image to be predicted: ')

        input_image = cv2.imread(image1_path)

        input_image_resized = cv2.resize(input_image, (128,128))

        input_image_scaled = input_image_resized / 255

        input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])

        input_prediction = model.predict(input_image_reshaped)

        input_pred_label = np.argmax(input_prediction)

        print("The input prediction result :",input_pred_label)


        # Calculate SSIM
        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray_img1, gray_img2, full=True)
        ans = []
        img = cv2.imread(image1_path)
            # get grayscale image
        def resize(image):
            return cv2.resize(image,(400,300))
        def get_grayscale(image):
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # noise removal
        def remove_noise(image):
            return cv2.medianBlur(image,5)
        
        #thresholding
        def thresholding(image):
            return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #adaptiveThresholding
        def adadaptiveThresholding(image):
            return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)

        #dilation
        def dilate(image):
            kernel = np.ones((5,5),np.uint8)
            return cv2.dilate(image, kernel, iterations = 1)
            
        #erosion
        def erode(image):
            kernel = np.ones((5,5),np.uint8)
            return cv2.erode(image, kernel, iterations = 1)

        #opening - erosion followed by dilation
        def opening(image):
            kernel = np.ones((5,5),np.uint8)
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        #canny edge detection
        def canny(image):
            return cv2.Canny(image, 100, 200)

        #skew correction
        def deskew(image):
            coords = np.column_stack(np.where(image > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated

        #cropping
        def cropping(image):
            return image[200:510,200:2200]

        #template matching
        def match_template(image, template):
            return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

        image = cv2.imread(image1_path)

        gray = get_grayscale(image)
        #cv2.imshow('gray',gray)
        thresh = thresholding(gray)
        #cv2.imshow('thresh',thresh)
        opening = opening(gray)
        #cv2.imshow('opening',opening)
        canny = canny(gray)
        #cv2.imshow('canny',canny)
        h, w, c = img.shape
        boxes = pytesseract.image_to_boxes(img) 
        myconfig = r"--oem 3 --psm 6"
        data = pytesseract.image_to_data(gray,config=myconfig, output_type=Output.DICT)
        text_data = data['text']
       

        def preprocess_finale(im):
            im= cv2.bilateralFilter(im, 5, 55,60)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            _, im = cv2.threshold(im, 240, 255, 1)
            return im

        #img = Image.open("Image.png")
        img = cv2.cvtColor(np.array(img2), cv2.COLOR_BGRA2BGR)

        custom_config = r"--oem 3 --psm 4 -c tessedit_char_whitelist= '0123456789. '"
        im=preprocess_finale(img)

        text = pytesseract.image_to_string(im, lang='eng', config=custom_config)
        print("Extracted Signature : ",text)
        for b in boxes.splitlines():
            b = b.split(' ')
        img = cv2.rectangle(gray, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
        d = pytesseract.image_to_data(gray,output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 80:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                img = cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for i in text_data:
            if(len(i)>2 and i.isascii()):
                ans.append(i)
        print("Extracted Information from cheque : ",ans)
        cv2.imshow('img', gray)
        cv2.waitKey(0)

        score  = calculate_ssim(image1_path,image2_path)
        print("Calculate SSIM score : ",abs(score))
        # Determine if images are forged based on SSIM score
        if input_pred_label == 1 :
            forged = False
            return jsonify({'result':"The given cheque is a original cheque"})
        elif input_pred_label == 0:
            forged = True
            return jsonify({'result': 'The given cheque is a fake cheque'})
        else: 
            return jsonify("Give a valid input cheque image")
        


# Home page with buttons
@app.route('/')
def home():
    return render_template('login.html')

# Image processing route
@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        operation = request.form['operation']
        f = request.files['image']
        filename = f.filename
        print(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        print(image_path)
        processed_image = process_image(image_path, operation)

        cv2.imwrite(r"static//processed_image.jpg", processed_image)
        return render_template('result.html', image_path=r"static//processed_image.jpg")

# Function to read CSV file
def read_csv():
    with open('users.csv', 'r') as file:
        reader = csv.DictReader(file)
        users = list(reader)
    return users

# Function to write to CSV file
def write_csv(users):
    with open('users.csv', 'w', newline='') as file:
        fieldnames = ['username', 'password', 'email', 'fullname','phonenumber']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(users)




@app.route('/uploadDataset', methods=['GET'])
def upload_dataset():
    global filename
    filename = "C:\\Users\\prave\\Final_Year_Project\\Dataset\\TrainSet\\X"
    return jsonify({'message': 'Dataset selected successfully'})


@app.route('/splitDataset', methods=['GET'])
def split_dataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    global pca
    X = np.load('features/X.txt.npy')
    Y = np.load('features/Y.txt.npy')
    X = np.reshape(X, (X.shape[0], (X.shape[1] * X.shape[2] * X.shape[3])))

    pca = PCA(n_components=100)
    X = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    return jsonify({'message': 'Dataset split successfully',
                    'total_images': len(X),
                    'train_split': len(X_train),
                    'test_split': len(X_test)})




# User registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        phonenumber = request.form['phonenumber']
        fullname = request.form['fullname']

        users = read_csv()
        users.append({'username': username, 'password': password, 'email': email, 'fullname': fullname, 'phonenumber':phonenumber})
        write_csv(users)
        users = read_csv()

        return redirect(url_for('login'))

    return render_template('register.html')

# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        

        users = read_csv()
        for user in users:
            if user['username'] == username and user['password'] == password:
                #return f"Welcome, {user['fullname']}!"
                return render_template('otp.html')

        return "Invalid username or password"

    return render_template('login.html')


# OTP generation 
otp = random.randint(1000,9999)
generated_otp = otp

@app.route('/getOTP',methods = ['POST'])
def getOTP():
    number = request.form['phonenumber']
    val = getOTPApi(number)
    if val:
        return render_template('otp.html')
# def generateOTP():
#     return random.randrange(100000,999999)
def getOTPApi(number):
    # otp = generateOTP()
    global msg 
    if(number == "7569993454"):
        account_id_1 = "ACb149e7a0d945b21d311a10f1e1c02a7b"
        auth_token_1 = 'd1c66b4f2384ee26098caa8ef846a492'
        client_1 = Client(account_id_1,auth_token_1)
        msg = client_1.messages.create(
            body = f"Your Online  Cheque Curation and Abstract System Verification Code: {otp}",
            from_ = "+19282491201",
            to = "+917569993454"
        )
    elif(number == "9949700759"):
        account_id_2 = "ACe254c3696c6e63e0f1585af6d173691f"
        auth_token_2 = '777a478fd79b98b597f6c8a985f85cf0'
        client_2 = Client(account_id_2,auth_token_2)
        msg = client_2.messages.create(
            body = f"Your Online  Cheque Curation and Abstract System Verification Code: {otp}",
            from_ = "+19792726652",
            to = "+919949700759"
        )
    elif(number == "8179040458"):
        account_id_3 = "ACdc90c033b749e1ff3c6c424af74d24f7"
        auth_token_3 = '21928393bee43cc0fcbaa75af8cd45b5'
        client_3 = Client(account_id_3,auth_token_3)
        msg = client_3.messages.create(
            body = f"Your Online  Cheque Curation and Abstract System Verification Code: {otp}",
            from_ = "+12135834640",
            to = "+918179040458"
        )
    elif(number == "9959615537"):
        account_id_4 = "AC675c8a4356802c994cca184f012021fb"
        auth_token_4 = 'b8ab36cc2f8f2f255bc5690eaf7bfa18'
        client_4 = Client(account_id_4,auth_token_4)
        msg = client_4.messages.create(
            body = f"Your Online  Cheque Curation and Abstract System Verification Code: {otp}",
            from_ = "++15169730416",
            to = "+919959615537"
        )
        
    if msg.sid:
        return True
    else:
        return False


@app.route('/validateOTP', methods = ['POST'])
def validateOTP():
    entered_otp = request.form['otp']
    if(str(generated_otp) == str(entered_otp)):
        return render_template('index1.html')
    else:
        return "Not a Verified User..."
    



if __name__ == '__main__':
    app.run(debug=True)
