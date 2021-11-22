import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    rect[0] = pts[2]
    rect[1] = pts[0]
    rect[2] = pts[1]
    rect[3] = pts[3]
    return rect

def show_in_notebook(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


class GoCv:

    def __init__(self, base_image_addr):
        self.base_image = cv2.imread(base_image_addr,1)

    def find_perspective_transform_data(self):
        img = self.base_image
        h, w, ch = img.shape

        green_low = np.array([0, 220, 0])
        green_high = np.array([10, 255, 10])
        mask = cv2.inRange(img, green_low, green_high)

        coord = cv2.findNonZero(mask)
        coord_xy = [[dot[0][0], dot[0][1]] for dot in coord]
        np_coord = np.array(coord_xy)

        kmeans = KMeans(n_clusters=4, random_state=0).fit(np_coord)

        pts = kmeans.cluster_centers_

        return pts

    def four_point_transform(self, pts, raw_img):
        rect = order_points(pts)

        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(raw_img, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    def construct_lookup_table(self):
        newbaseImg = self.four_point_transform(self.find_perspective_transform_data(), self.base_image)

        gray = cv2.cvtColor(newbaseImg, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 110, 320, apertureSize=3)
        show_in_notebook(edges)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 110)

        hori = []
        verti = []
        for line in lines:
            rho, theta = line[0]
            if theta <= 1.8 and theta >= 1.2:
                verti.append(line)
            else:
                hori.append(line)

        intersections = []
        for xline in hori:
            for yline in verti:
                intersections.append(intersection(xline, yline))

        from sklearn.cluster import KMeans
        X = np.array(intersections)
        kmeans = KMeans(n_clusters=169, random_state=0).fit(X)

        coord_array = np.array([[int(x[0]), int(x[1])] for x in kmeans.cluster_centers_])
        sorted_cord = coord_array[coord_array[:, 0].argsort()]
        coord_lookup = {}
        for i in range(13):
            row_sorted = sorted_cord[13 * i:13 * i + 13]
            rc_sorted = row_sorted[row_sorted[:, 1].argsort()]
            for j in range(13):
                coord_lookup[str(i) + ',' + str(j)] = rc_sorted[j]

        return coord_lookup









if __name__ == "__main__":
    go = GoCv('../img/highres_board.png')
