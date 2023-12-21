import os
import numpy as np
import cv2
from keras.models import load_model


def cut1(image, box):
    # 将box转换为四个顶点坐标
    pts = box.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")

    # 计算左上、右上、右下、左下四个顶点的坐标
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # 计算原始图像中车牌的宽度和高度
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后的四个顶点坐标
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # 计算透视变换矩阵，并应用到原始图像上
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def location_demo(img):
    image = cv2.resize(img,(1200,1200))    #对读取图片做大小调整
    lower_blue = np.array([100, 80, 60])     #设置蓝色和绿色色彩空间范围
    upper_blue = np.array([130, 255, 255])
    lower_green = np.array([30, 100, 100])
    upper_green = np.array([78, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   #转换色彩空间BGR转HSV
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)  #设置mask
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    output = cv2.bitwise_and(hsv, hsv, mask=mask_blue)#+mask_green )  #蓝色的车牌，采用蓝色mask
    Matrix = np.ones((20, 20), np.uint8)

    img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)

    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)

    # 根据阈值找到对应颜色
    img_edge3 = np.clip(img_edge2, 0, 255)  # 归一化也行

    contours, hierarchy = cv2.findContours(img_edge3[..., 0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 对图像进行轮廓检测
    for i in range(0, len(contours)):  #对所有检测到的轮廓进行按（车牌）面积筛选
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        #求轮廓面积
        if area >3000 :
            rect=cv2.minAreaRect(cnt)
            #对所选取的面积求矩形最小面积
            box = cv2.boxPoints(rect)
            #矩形点信息提取
            box = np.int0(box)
            cj = cut1(image, box)
            image = cv2.drawContours(image, [box], -1, (0, 255, ), 4)
            return cj

def my_cut(img_name):
    loc = [50, 80, 96, 136, 176, 216, 256, 296]  # 分割车牌字母位置
    img = cv2.imread(img_name)  # 读取分割图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转换色彩为灰度图像
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)  # 做滤波
    immean = cv2.mean(img)  # 求各分量均值，目的是什么车牌
    img_gray2 = cv2.equalizeHist(img_gray)  # 均衡图像

    if immean[0] > immean[1]:  # 绿色车牌和蓝色刚好相反
        binary_threshold = 190  # 设置阈值
    else:
        binary_threshold = 80

    # 2、按照前面设定阀值，将灰度图二值化
    img_thre = img_gray2
    cv2.threshold(img_gray2, binary_threshold, 255, cv2.THRESH_BINARY_INV, img_thre)
    img_thre = cv2.resize(img_thre, (300, 80))  # 改变图像大小到固定值
    cv2.imwrite('img80.jpg', img_thre)  # 这是已经处理好的黑白色车牌
    # 下面是把车牌每一个字母分割开来，可以自己训练字符集识别

    old = 0
    ind = 1

    for i in loc:
        cj = img_thre[1:80, old:i]

        image3 = cv2.resize(cj, (32, 40))
        cv2.bitwise_not(image3, image3)
        cv2.imwrite('character' + format(ind) + '.jpg', image3)
        old = i
        ind = ind + 1
    return ind


def horizontal_cut_chars(plate):
    char_addr_list = []
    area_left,area_right,char_left,char_right= 0,0,0,0
    img_w = plate.shape[1]

    # 获取车牌每列边缘像素点个数
    def getColSum(img,col):
        sum = 0
        for i in range(img.shape[0]):
            sum += round(img[i, col]/255)
        return sum

    sum = 0
    for col in range(img_w):
        sum += getColSum(plate,col)
    # 每列边缘像素点必须超过均值的60%才能判断属于字符区域
    col_limit = 0#round(0.5*sum/img_w)
    # 每个字符宽度也进行限制
    charWid_limit = [round(img_w/12),round(img_w/5)]
    is_char_flag = False

    for i in range(img_w):
        colValue = getColSum(plate,i)
        if colValue > col_limit:
            if is_char_flag == False:
                area_right = round((i+char_right)/2)
                area_width = area_right-area_left
                char_width = char_right-char_left
                if (area_width>charWid_limit[0]) and (area_width<charWid_limit[1]):
                    char_addr_list.append((area_left,area_right,char_width))
                char_left = i
                area_left = round((char_left+char_right) / 2)
                is_char_flag = True
        else:
            if is_char_flag == True:
                char_right = i-1
                is_char_flag = False
    # 手动结束最后未完成的字符分割
    if area_right < char_left:
        area_right,char_right = img_w,img_w
        area_width = area_right - area_left
        char_width = char_right - char_left
        if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
            char_addr_list.append((area_left, area_right, char_width))
    return char_addr_list


def get_chars(car_plate):
    img_h,img_w = car_plate.shape[:2]
    h_proj_list = [] # 水平投影长度列表
    h_temp_len,v_temp_len = 0,0
    h_startIndex,h_end_index = 0,0 # 水平投影记索引
    h_proj_limit = [0.2,0.8] # 车牌在水平方向得轮廓长度少于20%或多余80%过滤掉
    char_imgs = []

    # 将二值化的车牌水平投影到Y轴，计算投影后的连续长度，连续投影长度可能不止一段
    h_count = [0 for i in range(img_h)]
    for row in range(img_h):
        temp_cnt = 0
        for col in range(img_w):
            if car_plate[row,col] == 255:
                temp_cnt += 1
        h_count[row] = temp_cnt
        if temp_cnt/img_w<h_proj_limit[0] or temp_cnt/img_w>h_proj_limit[1]:
            if h_temp_len != 0:
                h_end_index = row-1
                h_proj_list.append((h_startIndex,h_end_index))
                h_temp_len = 0
            continue
        if temp_cnt > 0:
            if h_temp_len == 0:
                h_startIndex = row
                h_temp_len = 1
            else:
                h_temp_len += 1
        else:
            if h_temp_len > 0:
                h_end_index = row-1
                h_proj_list.append((h_startIndex,h_end_index))
                h_temp_len = 0

    # 手动结束最后得水平投影长度累加
    if h_temp_len != 0:
        h_end_index = img_h-1
        h_proj_list.append((h_startIndex, h_end_index))
    # 选出最长的投影，该投影长度占整个截取车牌高度的比值必须大于0.5
    h_maxIndex, h_maxHeight = 0, 0
    for i, (start, end) in enumerate(h_proj_list):
        if h_maxHeight < (end-start):
            h_maxHeight = (end-start)
            h_maxIndex = i
    if h_maxHeight/img_h < 0.5:
        return char_imgs
    chars_top,chars_bottom = h_proj_list[h_maxIndex][0],h_proj_list[h_maxIndex][1]

    plates = car_plate[chars_top:chars_bottom+1,:]
    # cv2.imwrite('car.jpg',car_plate)
    # cv2.imwrite('plate.jpg', plates)
    char_addr_list = horizontal_cut_chars(plates)

    for i,addr in enumerate(char_addr_list):
        char_img = car_plate[chars_top:chars_bottom+1,addr[0]:addr[1]]
        char_img = cv2.resize(char_img,(64,64))
        char_imgs.append(char_img)
    return char_imgs


def extract_char(car_plate):
    gray_plate = cv2.cvtColor(car_plate,cv2.COLOR_BGR2GRAY)

    ret,binary_plate = cv2.threshold(gray_plate,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    char_img_list = get_chars(binary_plate)
    i=0
    for img in char_img_list:
        cv2.imwrite('char' + format(i) + '.jpg', img)
        i=i+1
    return i


data_dir = "cnn_char_train"
classes = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]


def run(photo_path):
    # 1、读取图片
    imgread = cv2.imread(photo_path)
    frame = location_demo(imgread)  # 定位车牌，返回车牌
    cv2.imwrite("img_70.jpg", frame)  # 写入车牌文件

    img = cv2.imread("img_70.jpg")
    len = extract_char(img)
    loaded_model = load_model('model.h5')
    predicted_char = []
    for i in range(len):
        image = cv2.imread('char' + format(i) + '.jpg')
        # 对图像进行预处理，包括缩放、归一化和维度调整
        image = cv2.resize(image, (64, 64))  # 将图像调整为所需的尺寸
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0  # 归一化到 [0, 1] 范围
        image = np.expand_dims(image, axis=0)  # 增加批次维度
        res = loaded_model.predict(image)
        predicted_class_index = np.argmax(res)
        predicted_char.append(classes[predicted_class_index])
    print('Predicted character: ', predicted_char)
    print(type(predicted_char))
    number = ""
    for c in predicted_char:
        number += c
    return number
