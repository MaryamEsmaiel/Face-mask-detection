import gradio as gr
import cv2
from detect_faces_img import detect_face_img
from detect_faces_vid import detect_bounding_box


'''demo = gr.Interface(fn=detect_face_img,
                    inputs=gr.inputs.Image(type="pil"),
                    outputs='image')
demo.launch()'''

def detect_faces(vid):
    # loop for Real-Time Face Detection
    while True:

        video_capture = cv2.VideoCapture(0)
        result, video_frame = video_capture.read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully

        faces = detect_bounding_box(
            video_frame
        )  # apply the function we created to the video frame

        cv2.imshow(
            "My Face Detection Project", video_frame
        )  # display the processed frame in a window named "My Face Detection Project"

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


with gr.Blocks() as demo:
    gr.Markdown("detect faces in an image or video")
    img_button = gr.Button("detect_img")
    img_button.click(detect_face_img, inputs=gr.inputs.Image(type='pil'), outputs='image')

    vid_button = gr.Button("detect")
    vid_button.click(detect_faces, inputs='video', outputs='video')

demo.launch()