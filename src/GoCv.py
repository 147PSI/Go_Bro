import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[0]
    rect[1] = pts[2]
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
        self.base_image = cv2.imread(base_image_addr, 1)
        self.pts = self.find_perspective_transform_data()
        self.lookup_table = self.construct_lookup_table()

    def find_perspective_transform_data(self):
        img = self.base_image

        # *********************
        # show_in_notebook(img)
        # *********************

        green_low = np.array([0, 220, 0])
        green_high = np.array([10, 255, 10])
        mask = cv2.inRange(img, green_low, green_high)

        coord = cv2.findNonZero(mask)
        coord_xy = [[dot[0][0], dot[0][1]] for dot in coord]
        np_coord = np.array(coord_xy)

        kmeans = KMeans(n_clusters=4, random_state=0).fit(np_coord)

        pts = kmeans.cluster_centers_

        return order_points(pts)

    def four_point_transform(self, raw_img):

        (tl, tr, br, bl) = self.pts

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

        M = cv2.getPerspectiveTransform(self.pts, dst)
        warped = cv2.warpPerspective(raw_img, M, (maxWidth, maxHeight))

        # *********************
        # show_in_notebook(warped)
        # *********************

        # return the warped image
        return warped

    def construct_lookup_table(self):
        newbaseImg = self.four_point_transform(self.base_image)

        gray = cv2.cvtColor(newbaseImg, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 200, 320, apertureSize=3)

        # *********************
        # show_in_notebook(edges)
        # *********************

        lines = cv2.HoughLines(edges, 1, np.pi / 90, 110)

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

        X = np.array(intersections)
        kmeans = KMeans(n_clusters=169, random_state=0).fit(X)

        coord_array = np.array([[int(x[0]), int(x[1])] for x in kmeans.cluster_centers_])

        # *********************
        # vis6 = newbaseImg.copy()
        # for center in coord_array:
        #     vis6 = cv2.circle(vis6, [int(center[0]), int(center[1])], 4, (0, 0, 255), -1)
        # show_in_notebook(vis6)
        # *********************


        sorted_cord = coord_array[coord_array[:, 0].argsort()]
        coord_lookup = []
        for i in range(13):
            row_sorted = sorted_cord[13 * i:13 * i + 13]
            rc_sorted = row_sorted[row_sorted[:, 1].argsort()]
            x_row = []
            for j in range(13):
                x_row.append(rc_sorted[j])
            coord_lookup.append(x_row)
        coord_lookup = np.array(coord_lookup)

        return coord_lookup

    def mapping_image_to_list(self, new_img_add):
        new_base_img = cv2.imread(new_img_add, 1)

        warp_new_base = self.four_point_transform(new_base_img)

        # *********************
        # show_in_notebook(warp_new_base)
        # *********************

        step_gray = cv2.cvtColor(warp_new_base, cv2.COLOR_BGR2GRAY)
        step_gray = cv2.medianBlur(step_gray, 5)

        # *********************
        # show_in_notebook(step_gray)
        # *********************

        rows = step_gray.shape[0]
        circles = cv2.HoughCircles(step_gray, cv2.HOUGH_GRADIENT, 1, rows / 19,
                                   param1=80, param2=23,
                                   minRadius=10, maxRadius=30)
        if circles is not None:
            circles = np.uint16(np.around(circles))
        # *********************
        # vis7 = warp_new_base.copy()
        # for i in circles[0, :]:
        #     center = (i[0], i[1])
        #     # circle center
        #     cv2.circle(vis7, center, 1, (0, 100, 100), 3)
        #     # circle outline
        #     radius = i[2]
        #     cv2.circle(vis7, center, radius, (255, 0, 255), 3)
        # show_in_notebook(vis7)
        # *********************

        # *********************
        vis8 = warp_new_base.copy()
        # *********************

        r = 8
        mean_col = []
        for col in self.lookup_table:
            col_sum = sum(col[:, 0])
            mean_col.append(col_sum // 13)

        mean_row = []
        for i in range(13):
            row = self.lookup_table[:, i]
            row_sum = sum(row[:, 1])
            mean_row.append(row_sum // 13)

        stone_map = np.zeros((13, 13))

        for circle in circles[0, :]:
            col_val = circle[0]
            row_val = circle[1]

            row_index = -1
            col_index = -1

            for i in range(len(mean_row)):
                if abs(row_val - mean_row[i]) <= 15:
                    row_index = i
                    break
            for i in range(len(mean_col)):
                if abs(col_val - mean_col[i]) <= 15:
                    col_index = i
                    break

            roi = vis8[row_val - r: row_val + r, col_val - r: col_val + r]

            # *********************
            # show_in_notebook(roi)
            # *********************

            width, height = roi.shape[:2]
            mask = np.zeros((width, height, 3), roi.dtype)
            cv2.circle(mask, (int(width / 2), int(height / 2)), r, (255, 255, 255), -1)
            dst = cv2.bitwise_and(roi, mask)
            data = []
            for k in range(3):
                channel = dst[:, :, k]
                indices = np.where(channel != 0)[0]
                if len(indices) == 0:
                    data.append(0)
                else:
                    color = np.mean(channel[indices])
                    data.append(int(color))

            if sum(data) / len(data) < 50:
                stone_map[row_index][col_index] = 1
            elif sum(data) / len(data) > 200:
                stone_map[row_index][col_index] = 2
            else:
                stone_map[row_index][col_index] = 0

        return stone_map

if __name__ == "__main__":
    go = GoCv('../img/board/labeled.png')

    print(go.mapping_image_to_list("../img/board/step2.png"))
