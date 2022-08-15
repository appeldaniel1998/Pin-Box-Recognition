from roboflowoak import RoboflowOak
import cv2
import time
import numpy as np

'''
prerequisites of the program:
1. The box is towards us with the long side and with no angle to either side.
2. The box is far enough from the camera for the depth calculations to be correct (approx 50 cm).
3. The box is on a clear brown surface similar to the training images of the nn (otherwise does not recognize the box and pic correctly).
4. The box is 17 x 15 wholes, does not change. neither does the pin.
5. The program takes the box as full of wholes with no space between in the middle (as is the truth), but should not point to them. We assume 2 lines of wholes
    in the space.

true dimensions of box: 45x35 mm
distance between wholes - 2.5mm to each side
'''


if __name__ == '__main__':
    # instantiating an object (rf) with the RoboflowOak module
    rf = RoboflowOak(model="pin-box", confidence=0.4, overlap=0.5, version='2',
                     api_key="cNpdg5tikNEVov7CYRVN", rgb=True, depth=True, device=None, blocking=True)
    # Running our model and displaying the video output with detections
    depths = []
    widths = []
    while True:
        t0 = time.time()
        # The rf.detect() function runs the model inference
        result, frame, raw_frame, depth = rf.detect(visualize=True)
        predictions = result["predictions"]
        # {
        #    predictions:q
        #    [ {
        #        x: (middle),
        #        y:(middle),
        #        width:
        #        height:
        #        depth: ###->
        #        confidence:
        #        class:
        #        mask: {
        #    ]
        # }
        # frame - frame after preprocs, with predictions
        # raw_frame - original frame from your OAK
        # depth - depth map for raw_frame, center-rectified to the center camera

        # timing: for benchmarking purposes
        t = time.time() - t0

        box = None
        pin = None

        for i in predictions:
            if i.class_name == 'box':
                if box is None:
                    box = i
                else:
                    if box.confidence < i.confidence:
                        box = i
            elif i.class_name == 'pin':
                if pin is None:
                    pin = i
                else:
                    if pin.confidence < i.confidence:
                        pin = i

        if box is not None and pin is not None and 0 < abs(predictions[0].depth - predictions[1].depth) < 10:
            # print("INFERENCE TIME IN MS ", 1 / t)
            # print("PREDICTIONS ", [p.json() for p in predictions])

            print()
            print(box.json())
            print(pin.json())

            if pin.depth - box.depth != np.inf:
                depths.append(pin.depth - box.depth)
                print((pin.depth - box.depth) / 2.5)
                # print(round(np.mean(depths) / 2.5))

            # if pin.depth - box.depth != np.inf:
            widths.append(abs(box.x - pin.x))
            print(abs(box.x - pin.x) / 2.5)
            # print(round(np.mean(widths) / 2.5))





        # setting parameters for depth calculation
        max_depth = np.amax(depth)
        cv2.imshow("depth", depth / max_depth)
        # displaying the video feed as successive frames
        cv2.imshow("frame", frame)



        # how to close the OAK inference window / stop inference: CTRL+q or CTRL+c
        if cv2.waitKey(1) == ord('q'):
            break
