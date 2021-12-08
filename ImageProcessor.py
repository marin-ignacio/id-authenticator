import numpy as np
import cv2
import imutils


class ImageProcessor:

    def __init__(self):
        pass

    def readImage(self, filePath: str) -> np.ndarray:
        """ This function takes the path of an image file and loads their
            information into a matrix.

        Args:
            filePath (str): image file path.

        Returns:
            np.ndarray: output image.
        """
        return cv2.imread(filePath)

    def imResize(self, img: np.ndarray, width: int) -> np.ndarray:
        """ This function takes an image and modify their size, without
            loosing the aspect ratio.

        Args:
            img (np.ndarray): input image.
            width (int): new width for the image.

        Returns:
            np.ndarray: output image.
        """
        return imutils.resize(image=img,
                              width=width)

    def imScale(self, img: np.ndarray, scale: tuple) -> np.ndarray:
        """ This function takes an image and scales its size.

        Args:
            img (np.ndarray): input image.
            scale (tuple): scale factor along each axis (x, y).

        Returns:
            np.ndarray: output image.
        """
        return cv2.resize(src=img,
                          dsize=None,
                          fx=scale[0],
                          fy=scale[1],
                          interpolation=cv2.INTER_CUBIC)

    def im2grayscale(self, img: np.ndarray) -> np.ndarray:
        """ This function takes and RGB image and converts it into
            a grayscale image.

        Args:
            img (np.ndarray): input image.

        Returns:
            np.ndarray: output image.
        """
        return cv2.cvtColor(src=img,
                            code=cv2.COLOR_BGR2GRAY)

    def im2bin(self, img: np.ndarray, thresh: int = 0, maxval: int = 255) -> np.ndarray:
        """ This function takes an image and converts it into a binary image.

        Args:
            img (np.ndarray): input image.
            thresh (int, optional): threshold value. Defaults to 0.
            maxval (int, optional): maximum value to use. Defaults to 255.

        Returns:
            np.ndarray: output image.
        """
        return cv2.threshold(src=img,
                             thresh=thresh,
                             maxval=maxval,
                             type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def im2binAdaptive(self, img: np.ndarray, maxValue: int, blockSize: int, C: int) -> np.ndarray:
        """ This function takes an image and converts it into a binary image
            using an adaptive thresholding.

        Args:
            img (np.ndarray): input image.
            maxValue (int): threshold value.
            blockSize (int): size of the neighborhood area.
            C (int): constant to subtract from the weighted mean calculated.

        Returns:
            np.ndarray: output image.
        """
        return cv2.adaptiveThreshold(src=img,
                                     maxValue=maxValue,
                                     adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                     thresholdType=cv2.THRESH_BINARY,
                                     blockSize=blockSize,
                                     C=C)

    def reduceNoise(self, img: np.ndarray, ksize: tuple = (3, 3), sigmaX: int = 0) -> np.ndarray:
        """ This function takes an image and reduces the noise present in it by using a
            Gaussian filter.

        Args:
            img (np.ndarray): input image.
            ksize (tuple, optional): size of the neighborhood area. Defaults to (3, 3).
            sigmaX (int, optional): standard deviation of the Gaussian kernel. Defaults to 0.

        Returns:
            np.ndarray: output image.
        """
        return cv2.GaussianBlur(src=img,
                                ksize=ksize,
                                sigmaX=sigmaX)

    def blackHatTransform(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """ This funcion takes an image and applies to it and black-hat tranform to it.

        Args:
            img (np.ndarray): input image.
            kernel (np.ndarray): structuring element.

        Returns:
            np.ndarray: output image.
        """
        return cv2.morphologyEx(src=img,
                                op=cv2.MORPH_BLACKHAT,
                                kernel=kernel)

    def detectEdges(self, img: np.ndarray, dx: int, dy: int, ksize: int) -> np.ndarray:
        """ This function takes an image and applies to it an edge detection algorithm
            using a Sobel operator.

        Args:
            img (np.ndarray): input image.
            dx (int): dx order of the derivative x.
            dy (int): dy order of the derivative y.
            ksize (int): size of the extended Sobel kernel; it must be 1, 3, 5, or 7.

        Returns:
            np.ndarray: output image.
        """
        gradX = cv2.Sobel(src=img,
                          ddepth=cv2.CV_32F,
                          dx=dx,
                          dy=dy,
                          ksize=ksize)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        return gradX

    def morphClose(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """ This function takes an image and applies to it a morphological
            closing operation.

        Args:
            img (np.ndarray): input image.
            kernel (np.ndarray): structuring element.

        Returns:
            np.ndarray: output image.
        """
        return cv2.morphologyEx(src=img,
                                op=cv2.MORPH_CLOSE,
                                kernel=kernel)

    def morphOpen(self, img: np.ndarray, kernel: np.ndarray, iters: int = 1) -> np.ndarray:
        """ This function takes an image and applies to it a morphological
            opening operation.

        Args:
            img (np.ndarray): input image.
            kernel (np.ndarray): structuring element.
            iters (int): number of times erosion and dilation are applied. Defaults to 1.

        Returns:
            np.ndarray: output image.
        """
        return cv2.morphologyEx(src=img,
                                op=cv2.MORPH_OPEN,
                                kernel=kernel,
                                iterations=iters)

    def findContours(self, img: np.ndarray) -> np.ndarray:
        """ This function takes an image and, identifies and returns the contours
            of image segments.

        Args:
            img (np.ndarray): input image.

        Returns:
            np.ndarray: detected contours.
        """
        cnts = cv2.findContours(image=img,
                                mode=cv2.RETR_EXTERNAL,
                                method=cv2.CHAIN_APPROX_SIMPLE)

        return sorted(imutils.grab_contours(cnts),
                      key=cv2.contourArea,
                      reverse=True)

    def extractROI(self, img: np.ndarray, segment: tuple) -> np.ndarray:
        """ This function takes an image and extract the specified region of interest.

        Args:
            img (np.ndarray): input image.

        Returns:
            np.ndarray: output image.
        """
        x1, y1, x2, y2 = segment

        return img[y1:y2, x1:x2]

    def drawSegments(self, img: np.ndarray,
                     segments: list,
                     color: tuple = (255, 0, 0),
                     thickness: int = 1) -> np.ndarray:
        """ This function takes an image and draws all the segments contained
            in the given list.

        Args:
            img (np.ndarray): input image.
            segments (list): segments to draw.
            color (tuple, optional): segment color. Defaults to (255, 0, 0).
            thickness (int, optional): segment thickness. Defaults to 1.

        Returns:
            np.ndarray: output image.
        """
        # Goes through the all the segments
        for i in range(len(segments)):
            img = self.drawSegment(img=img, segment=segments[i])

        return img

    def drawSegment(self, img: np.ndarray,
                    segment: list,
                    color: tuple = (255, 0, 0),
                    thickness: int = 1) -> np.ndarray:
        """ This function takes an image and draws in it the given segment.

        Args:
            img (np.ndarray): input image.
            segment (list): segment to draw.
            color (tuple, optional): segment color. Defaults to (255, 0, 0).
            thickness (int, optional): segment thickness. Defaults to 1.

        Returns:
            np.ndarray: output image.
        """
        x1, y1, x2, y2 = segment
        return cv2.rectangle(img=img,
                             pt1=(x1, y1),
                             pt2=(x2, y2),
                             color=color,
                             thickness=thickness)
