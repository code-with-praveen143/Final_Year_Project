from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import numpy as np
import sklearn
##from sklearn import svm
##from sklearn.metrics import accuracy_score
##from sklearn.model_selection import train_test_split
##from sklearn.decomposition import PCA
##from sklearn.cluster import KMeans
import cv2
import os
##import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import requests
import csv
from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

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

        # Determine if images are forged based on SSIM score
        if score < 0.9:
            forged = True
        else:
            forged = False

        # Delete uploaded images
        

        return jsonify({'forged': forged, 'ssim': score})


# Home page with buttons
@app.route('/')
def home():
    return render_template('index1.html')

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
    filename = "C:\\Users\\VN Technologies\\Desktop\\website\\Dataset"
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


@app.route('/executeSVM', methods=['GET'])
def execute_svm():
    global classifier
    global svm_acc
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    svm_acc = accuracy_score(y_test, predict) * 100
    classifier = cls
    return jsonify({'message': 'SVM executed successfully', 'svm_accuracy': svm_acc})


@app.route('/executeKmeans', methods=['GET'])
def execute_kmeans():
    global kmeans_acc
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X_train)
    predict = kmeans.predict(X_test)
    kmeans_acc = accuracy_score(y_test, predict) * 100
    return jsonify({'message': 'K-Means executed successfully', 'kmeans_accuracy': kmeans_acc})


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
                return render_template('index1.html')

        return "Invalid username or password"

    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
