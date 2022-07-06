import cv2

img = cv2.imread(r"1.png", cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3,), (-1, -1))
dst1 = cv2.morphologyEx(src=img, op=cv2.MORPH_OPEN, kernel=kernel)
# dst = cv2.threshold(dst1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", dst1)
cv2.waitKey(0)
cv2.destroyWindow(img)
