from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from PIL import Image
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import requests
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
#from sklearn.decomposition import PCA 
#from sklearn.model_selection import train_test_split
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
        data = pytesseract.image_to_data(img,config=myconfig, output_type=Output.DICT)
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
        print(text)
        for b in boxes.splitlines():
            b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
        d = pytesseract.image_to_data(img,config=myconfig,output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 80:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        for i in text_data:
            if(len(i)>2 and i.isascii() ):
                ans.append(i)
        print(ans)
        # Determine if images are forged based on SSIM score
        if score < 0.9:
            forged = False
        else:
            forged = True
            

        # Delete uploaded images
        
        return jsonify({'forged': forged, 'ssim': score, 'text':ans})


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
        fieldnames = ['username', 'password', 'email', 'fullname']
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

def predictsign(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (64, 64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(64, 64, 3)
    im2arr = im2arr.astype('float32')
    im2arr = im2arr / 255
    test = []
    test.append(im2arr)
    test = np.asarray(test)
    test = np.reshape(test, (test.shape[0], (test.shape[1] * test.shape[2] * test.shape[3])))
    test = pca.transform(test)
    predict = classifier.predict(test)[0]
    msg = ''
    if predict == 0:
        msg = "not forged"
    if predict == 1:
        msg = "forged"
    dict['msg']=msg 
    img = cv2.imread(filename)
    img = cv2.resize(img, (400, 400))
    cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(msg, img)
    cv2.waitKey(0) 
    # closing all open windows 
    cv2.destroyAllWindows() 
    return msg, img





# User registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        fullname = request.form['fullname']

        users = read_csv()
        users.append({'username': username, 'password': password, 'email': email, 'fullname': fullname})
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
    if(number == "7569993454"):
        account_id_1 = "ACb149e7a0d945b21d311a10f1e1c02a7b"
        auth_token_1 = 'd1c66b4f2384ee26098caa8ef846a492'
        client_1 = Client(account_id_1,auth_token_1)
        msg = client_1.messages.create(
            body = f"Your Online Curation Cheque System Verification Code: {otp}",
            from_ = "+19282491201",
            to = "+917569993454"
        )
    elif(number == "9949700759"):
        account_id_2 = "ACe254c3696c6e63e0f1585af6d173691f"
        auth_token_2 = '777a478fd79b98b597f6c8a985f85cf0'
        client_2 = Client(account_id_2,auth_token_2)
        msg = client_2.messages.create(
            body = f"Your Online Curation Cheque System Verification Code: {otp}",
            from_ = "+19792726652",
            to = "+919949700759"
        )
    elif(number == "8179040458"):
        account_id_3 = "ACdc90c033b749e1ff3c6c424af74d24f7"
        auth_token_3 = '21928393bee43cc0fcbaa75af8cd45b5'
        client_3 = Client(account_id_3,auth_token_3)
        msg = client_3.messages.create(
            body = f"Your Online Curation Cheque System Verification Code: {otp}",
            from_ = "+12135834640",
            to = "+918179040458"
        )
    elif(number == "9959615537"):
        account_id_4 = "AC675c8a4356802c994cca184f012021fb"
        auth_token_4 = 'b8ab36cc2f8f2f255bc5690eaf7bfa18'
        client_4 = Client(account_id_4,auth_token_4)
        msg = client_4.messages.create(
            body = f"Your Online Curation Cheque System Verification Code: {otp}",
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
