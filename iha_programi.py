import torch
import cv2 as cv
from header import yolov5_path, model_source, video_source, logo_source, searching_for_uav, chasing_uav, uav_detected, kirmizi, mavi, yesil, beyaz, siyah


def program(capture, model, hedef_vurus_alani):
    while True:
        isTrue, frame = capture.read()
        result = model(frame)

        # bulunan uçakların merkezine çizgi çiz
        ucaklar = result.pandas().xywh[0]
        cv.rectangle(frame, hedef_vurus_alani[0], hedef_vurus_alani[1], kirmizi, 10) # Hedef vuruş alanını çiz
        
        for ucak in range(ucaklar.__len__()):
            a = ucaklar[ucak:ucak+1]
            if (float(a['confidence'][0:1]) > .7): # Eğer uçak olduğuna yüzde x eminse
                cv.line(frame,
                        (int(a['xcenter'][0:1]), int(a['ycenter'][0:1])), (640, 360), (255,0,0), 2)
                cv.rectangle(frame, (int(a['xcenter'][0:1]) - int(a['width'][0:1])//2, int(a['ycenter'][0:1]) - int(a['height'][0:1])//2), (int(a['xcenter'][0:1]) + int(a['width'][0:1])//2, int(a['ycenter'][0:1]) + int(a['height'][0:1])//2), kirmizi, 3) # Hedef vuruş alanını çiz
                
                
        if ucaklar.__len__()>0:
            cv.putText(frame, chasing_uav, (10,360), cv.FONT_HERSHEY_SIMPLEX, 1, beyaz, 1, cv.LINE_AA,False)
        else:
            cv.putText(frame, searching_for_uav, (10,360), cv.FONT_HERSHEY_SIMPLEX, 1, beyaz, 1, cv.LINE_AA,False)
        cv.imshow('video', frame)
        
        if cv.waitKey(1) & 0xFF==ord('d'):
            break


def end(capture):
    capture.release()
    cv.destroyAllWindows()


def load_video(video_source):
    return cv.VideoCapture(video_source)

def rectangles_and_logo(capture):
    logo = cv.imread(logo_source)
    isTrue, frame = capture.read()
    dikey, yatay, renk_kanali = frame.shape
    kamera_gorus_alani = yatay, dikey 
    hedef_vurus_alani = (
        (kamera_gorus_alani[0]//4, kamera_gorus_alani[1]//10),
        (kamera_gorus_alani[0] - kamera_gorus_alani[0]//4, kamera_gorus_alani[1] - kamera_gorus_alani[1]//10))
    kilitlenme_dortgeni = kamera_gorus_alani[0]//20, kamera_gorus_alani[1]//20,
    return hedef_vurus_alani

def load_model():
    model = torch.hub.load(yolov5_path,
                           'custom',
                           path = model_source,
                           source = 'local')
    model.cuda(0)
    return model

def main():
    model = load_model()
    capture = load_video(video_source)
    hedef_vurus_alani = rectangles_and_logo(capture)
    program(capture, model, hedef_vurus_alani)
    end(capture)
    
if __name__ == '__main__':
    main()
