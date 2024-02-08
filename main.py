import cv2

cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
line_y = 200
ob_count = 0

min_box_size = 500

def check_box_size(gray, x, y):
    width = 0
    max_width = gray.shape[1] - x
    while width < max_width and gray[y, x + width] < 127:
        width += 1
    height = 0
    max_height = gray.shape[0] - y
    while height < max_height and gray[y + height, x] < 127:
        height += 1

    box_area = width * height
    return box_area >= min_box_size

while True:
    ret, frame = cap.read()

    fgmask = bg_subtractor.apply(frame)

    thresh = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        extent = cv2.contourArea(cnt) / (w * h)

        if 0.8 < aspect_ratio < 1.2 and extent > 0.7:
            filtered_contours.append(cnt)

    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box_area = w * h
        if box_area >= min_box_size:
            if y < line_y < y + h:
                ob_count += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)
    cv2.putText(frame, f"Count: {ob_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('inventoryCounter', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
