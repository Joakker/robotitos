import os.path

import cv2 as cv

import numpy as np
import math



#a simple function to test different butter images I have saved. I'ts not actually needed
def get_image(n):
    path = "/home/felicia/Desktop/robot_stuff/butterbot/butter_pictures/"
    filename = "butter"+str(n)+".jpeg"
    path = path+filename
    os.path.exists(path)
    img = cv.imread(path)
    assert img is not None, "file could not be read, check with os.path.exists()"

    return img
#average color of a contour, in hsv
def contour_mean(image,contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, (255), thickness=cv.FILLED)
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return  cv.mean(image, mask=mask)


#Calculate the angle at pt2 formed by (pt1, pt2, pt3). For the spike removing thing (i got this from chatgpt)
def calculate_angle(pt1, pt2, pt3):
    vec1 = pt1 - pt2
    vec2 = pt3 - pt2
    dot_product = np.dot(vec1.flatten(), vec2.flatten())
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if (norm1 * norm2) == 0: return 0
    cos_angle = dot_product / (norm1 * norm2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip to avoid floating point issues
    return np.degrees(angle)

#recieves the polygon thing aproxPolyDP returns, and removes vertices with sharp angles
def remove_spikes(polygon):
    answer = []
    for i in range(len(polygon)):
        p1 = polygon[i - 1]
        p2 = polygon[i]
        p3 = polygon[(i + 1) % len(polygon)]
        if calculate_angle(p1,p2,p3)>50:
            answer.append(p2)

    return np.array(answer)

#returns how much butter-like a contour is
def butteriness(image_hsv,contour):
    ideal_butter_ratio = 1.7
    ideal_butter_color = [60,100,210] #hsv
    score = 0
    mean = contour_mean(image_hsv,contour)
    color_distance = 1
    color_distance += abs(ideal_butter_color[0]-mean[0])*3 #hue, very important
    color_distance += abs(ideal_butter_color[1]-mean[1])*0.5 #saturation is not that important methinks
    color_distance += abs(ideal_butter_color[2]-mean[2])*2 #lightness, yeh, important i guess
    score += 10000/color_distance #this keeps the value added by color distance between about 50-200
    #shape
    _, widhe, _ = cv.minAreaRect(contour)
    w=widhe[0]
    h=widhe[1]
    if w==0 or h==0: return 0 #honestly how would something so flat make it this far?
    ratio = max(w,h)/min(w,h)
    #what's the ideal butter ratio? eh, about 1.7
    ratio_error = abs(ratio-ideal_butter_ratio)**2+1 #add 1 to avoid 0
    score += 70/ratio_error #cases: 1.7->100 1->67 4->16
    return score
#aproximates a contour to a polygon
def polygonize(contour):
    epsilon = 0.005* cv.arcLength(contour,True)
    polygon = cv.approxPolyDP(contour,epsilon,True)
    return polygon

#I can't believe cv doesn't have something to convert a single color. I have to create a 1x1 image to convert it? I must be wrong
def hsv_to_rgb(hue, saturation, value):
    hsv_color = np.uint8([[[hue, saturation, value]]])
    rgb_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)
    return rgb_color[0][0]

def apply_tint(image, tint_color, intensity=0.2):
    tint = np.full_like(image, tint_color)#make an image of just the tint color
    # Blend the tint with the original image
    tinted_image = cv.addWeighted(image, 1.0, tint, intensity,0.1)
    return tinted_image


def sharpen_image(image):
    # apparently this kernel sharpens it
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv.filter2D(image, -1, kernel)


def increase_saturation(image, amount=1.5):
    # Convert the image to HSV color space
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Split the channels
    h, s, v = cv.split(hsv)

    # Multiply the saturation by the desired amount (e.g., 1.5 for 50% increase)
    s = cv.multiply(s, amount)

    # Clip the values to ensure they stay in valid range [0, 255]
    s = np.clip(s, 0, 255).astype(np.uint8)

    # Merge the channels back together
    hsv_enhanced = cv.merge([h, s, v])

    # Convert back to BGR color space
    return cv.cvtColor(hsv_enhanced, cv.COLOR_HSV2BGR)

def analyze(image):
    original_image = image
    #resize, we don't need all those pixels
    image = cv.resize(image,(300,300),interpolation = cv.INTER_LINEAR)

    #blur it a little, so it's a bit smoother
    #bilateral filter keeps those edges, well, edgy
    image = cv.bilateralFilter(image, 9,30,100)

    #get the part that's not not butter colored
    hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    color_mask = cv.inRange(hsvImage, np.array([25, 0, 50]), np.array([35, 230, 255]))
    color_mask2 = cv.inRange(hsvImage, np.array([0, 0, 220]), np.array([255, 20, 255]))
    color_mask3 = cv.inRange(hsvImage, np.array([20, 0, 50]), np.array([40, 140, 255]))
    color_mask2 = cv.bitwise_or(color_mask2,color_mask3)
    image = cv.bitwise_and(image, image, mask=cv.bitwise_or(color_mask,color_mask2))

    #sharpen
    #image = sharpen_image(image)

    #Ooh, I'm edging

    grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #clathe, to increase contrast

    clahe = cv.createCLAHE(clipLimit=5)
    clahe.apply(grey_image,grey_image)

    edges = cv.Canny(grey_image, 0, 500)
    edges = cv.dilate(edges, np.ones((4, 4), np.uint8), iterations=1)
    edges = cv.Canny(edges, 0, 500)
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    #edges = lineify(edges)
    #cv.imshow("e", edges)
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    #polygonize contous
    filtered_contours = [remove_spikes(polygonize(contour)) for contour in contours]
    #remove empty contours
    filtered_contours = [contour for contour in filtered_contours if contour.size>0]
    #filter out puny stuff
    filtered_contours = [contour for contour in filtered_contours if cv.contourArea(contour)>300]

    if len(filtered_contours)<=0: return None

    #butter_contour will be the butteriest of the contours, not guaranteed to actually be butter
    butter_contour = filtered_contours[0]
    hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    maxbutt = butteriness(hsvImage,butter_contour)
    for c in filtered_contours:
        if butteriness(hsvImage,c) > maxbutt:
            maxbutt = butteriness(hsvImage,c)
            butter_contour = c
    #get butter position on the screen
    M = cv.moments(butter_contour)
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    else:
        # In case of zero division, set a default position
        cX, cY = butter_contour[0][0][0], butter_contour[0][0][1]
    value_threshold = 100
    if maxbutt>value_threshold:
        return [cX/300,cY/300]
    else:
        return None

class PossibleButter:
    position = [0,0]
    permanenceScore = 0
    max_score = 200
    momentum = [0,0]
    def __init__(self,pos, initialScore=100):
        self.position = pos
        self.permanenceScore = initialScore
        self.momentum = [0,0]
    def increaseScore(self, amount):
        self.permanenceScore += amount
        if(self.permanenceScore>self.max_score):
            self.permanenceScore = self.max_score
    def decreaseScore(self):
        self.permanenceScore -= 1
    def mightBeMe(self,other):
        wasItMe = False
        p = self.position
        m = self.momentum
        estimatedNextPosition = [p[0]+m[0],p[1]+m[1]]
        dx = abs(estimatedNextPosition[0] - other.position[0])
        dy = abs(estimatedNextPosition[1] - other.position[1])
        dis = math.sqrt(dx**2+dy**2)
        if dis<0.1:
            newpos = [(p[0]+other.position[0])/2,(p[1]+other.position[1])/2]
            self.momentum = [newpos[0]-p[0],newpos[1]-p[1]]
            self.position = newpos
            wasItMe = True
        return wasItMe


class ButterDetector:
    possible_butters = []
    def receive_frame(self,frame):
        for b in self.possible_butters:
            b.decreaseScore()
            if b.permanenceScore<0:
                self.possible_butters.remove(b)
        butt_pos = analyze(frame)
        if butt_pos is None: return
        new_butt = PossibleButter(butt_pos)
        for b in self.possible_butters:
            if b.mightBeMe(new_butt):
                b.increaseScore(10)
                return
        self.possible_butters.append(new_butt)
    #returns the position of the butter. None if there isn't butter
    #note the coordinates are between 0 and 1
    def get_butter(self):
        ans = PossibleButter(None,-1)
        for b in self.possible_butters:
            if b.permanenceScore > ans.permanenceScore:
                ans = b
        if ans.permanenceScore>10:
            print(ans.permanenceScore)
            return ans.position
        else:
            return None




def main():
    buttDetector = ButterDetector()
    vid = cv.VideoCapture(2)
    if not vid.isOpened():
        print("Cannot open webcam")
        exit()
    while True:
        ret,frame = vid.read()
        buttDetector.receive_frame(frame)
        butter = buttDetector.get_butter()
        print(str(butter[0])+" "+str(butter[1]))
        if cv.waitKey(1) == ord('q'):
            break
    vid.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()