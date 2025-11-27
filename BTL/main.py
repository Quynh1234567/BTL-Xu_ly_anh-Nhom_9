
############################ KHAI BAO THU VIEN #########################################################################

import cv2
import numpy as np
import utlis
 
############################ CAU HINH CHUONG TRINH #####################################################################

webCamFeed = False
pathImage = "Picture/3.jpg"
cap = cv2.VideoCapture(2)
cap.set(10,160) # Chinh Độ sáng cho webcam
heightImg = 768
widthImg  = 512

########################################################################################################################

 # khoi tao thanh trượt điều chỉnh tham số
utlis.initializeTrackbars()
count=0 # biến đếm số file đã lưu

######################### TIEN XU LY ANH ###############################################################################

while True:
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8)
    # imgRed = np.zeros((heightImg, widthImg, 3), np.uint8)
    # imgRed[:] = (0, 0, 255)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    thres=utlis.Threshold()
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
 
################## TIM TAT CA COUNTOURS ################################################################################

    imgContours = img.copy()
    contours,_= cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

 
################## TIM BIGGEST COUNTOUR ################################################################################

    imgBigContour = img.copy()
    biggest = utlis.biggestContour(contours)
    if biggest.size != 0:
        biggest=utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) 
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))


################## XOA 20 PIXEL TU MOI CANH #####################################################################

        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,5) #trung vi

        # Image Array for Display
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])
 
    else:
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])
 
####################### LABELS FOR DISPLAY##############################################################################
    #lables = []
    lables = [["Img", "Img Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]
 
    stackedImage = utlis.stackImages(imageArray,0.5, lables)
    cv2.imshow("Result",stackedImage)

 
###################### SAVE IMAGE WHEN 's' key is pressed###############################################################
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("kq/myImage"+str(count)+".jpg",imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)   # Cho phép phóng to
        cv2.resizeWindow("Result", 1200, 900)          # Đặt kích thước ban đầu

        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1