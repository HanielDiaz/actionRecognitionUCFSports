import numpy as np
import cv2
import os
import torch
import pandas as pd
from skimage import io, transform
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.ion()

class FaceLandmarksDataset():
    def __init__(self,csv_file,root_dir,transform=None):
        """
        Args:
            csv_file(string): Path to the csv file with annotations.
            root_dir(string): Directory with all the images.
            transform(callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

def lucasKanadOpticalFlow(video):
    cap = cv2.VideoCapture(video)


    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=10,
                        qualityLevel=0.01,
                        minDistance=10,
                        blockSize=10)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                    maxLevel=6,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv2.destroyAllWindows()
    cap.release()


def dataloading(filename):
    # Read in a video file.
    cap = cv2.VideoCapture(filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('blackAndWhite.avi',fourcc,20.0,(640,480))
    # Read the first frame of the video.
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            # Convert it to black and white ("gray").
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.write(frame)
            # Output
            cv2.imshow('frame',gray_frame)
            if cv2.waitKey(400) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def dataLoadingTutorial(n):
    landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
    img_name = landmarks_frame.iloc[n,0]
    landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
    landmarks = landmarks.astype('float').reshape(-1,2)

    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))
    face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',root_dir='data/faces/')
    fig = plt.figure()
    for i  in range(len(face_dataset)):
        sample = face_dataset[i]

        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1,4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break
    plt.waitforbuttonpress()


def main():
    if(len(sys.argv) > 1):
        filename = sys.argv[1]
        # dataloading(filename)
        lucasKanadOpticalFlow(filename)
    else:
        image = 62
        dataLoadingTutorial(image)
    return 0


main()
