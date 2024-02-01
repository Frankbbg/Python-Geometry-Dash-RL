import cv2 as cv
import numpy as np
import mss

class PreprocessImage:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
    
    @staticmethod
    def preprocess_frame(self, frame, dimensions=(None, None)):
        # Convert frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        if dimensions[0] is not None and dimensions[1] is not None:
            # Resize / downsample the image
            resized = cv.resize(gray, (dimensions[0], dimensions[1]), interpolation=cv.INTER_AREA)
        else:
            resized = cv.resize(gray, (self.image_width, self.image_height), interpolation=cv.INTER_AREA)
        
        # Normalize the image
        normalized = resized / 255.0
        
        return normalized

    @staticmethod
    def stack_frames(self, stacked_frames, new_frame, is_new_episode, stack_size=4):
        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = np.zeros((self.image_width, self.image_height, stack_size))
            
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames = np.append(new_frame[..., np.newaxis], stacked_frames[:, :, :3], axis=2)

        return stacked_frames

class GDScreenCapture:
    def __init__(self, screenWidth, screenHeight):
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        
    def capture_screenshot(self, filename="debug.png", *, debug=False):
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": self.screenWidth, "height": self.screenHeight}
            img = sct.grab(monitor)
            
            if debug:
                # save to file
                cv.imwrite(filename, np.array(img))
                print(f"Screenshot saved to {filename}")
                
            img = PreprocessImage.preprocess_frame(img, (self.screenWidth, self.screenHeight))
            
            
            return img