import cv2
import constants
import model_fn
import utility_fn


def main():
    loaded_model = model_fn.load_saved_model('model1.h5')
    detector = cv2.CascadeClassifier(constants.CASCADE_PATH)
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        try:
            (check, frame) = capture.read()
            (frame_clone, gray_frame) = utility_fn.preprocess_frame(frame)

            rois = utility_fn.detect_face(detector, gray_frame)

            for roi in rois:
                (label, score) = model_fn.predict_image(loaded_model, roi["roi"])
                utility_fn.display_roi(frame_clone, label, score, roi)
            
            cv2.imshow("Face", frame_clone)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                capture.release()
                cv2.destroyAllWindows()
                break

        except(KeyboardInterrupt):
            capture.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
