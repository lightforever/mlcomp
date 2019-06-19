from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import os


class ImgProcessor:
    """
    Image processing information
    """
    min_x_dist = 8
    max_x_dist = 67
    anode_color = (29, 230, 181)
    cathode_color = (36, 28, 237)

    def __init__(self, img_orig, cathode_count, broken_threshold=0.5, cathode_threshold=0.5, anode_threshold=0.5):
        self.img_orig = img_orig
        self.img_draw = img_orig.copy() if img_orig is not None else None
        self.cathode_points = []
        self.anode_points = []
        self.broken_threshold = broken_threshold
        self.cathode_threshold = cathode_threshold
        self.anode_threshold = anode_threshold
        self.cathode_count = cathode_count

    def broken_report(self):
        """
        create broken report
        :return: dict with broken report
        """
        result = {
            'pairs': [],
            'pairs_broken': [],
            'prob_broken': 0
        }

        for crop in self.find_pairs():
            p = crop['prob_broken']
            result['prob_broken'] = max(result['prob_broken'], p)
            pair = {'cathode': crop['cathode'],
                    'anode': crop['anode'],
                    'side': crop['side'],
                    'dist': crop['dist'],
                    'cathode_point': crop['cathode_point'],
                    'anode_point': crop['anode_point'],
                    'prob_broken': p,
                    'cathode_w': crop['cathode_w'],
                    'anode_w': crop['anode_w'],
                    }
            result['pairs'].append(pair)
            if p >= self.broken_threshold:
                result['pairs_broken'].append(pair)

        result['broken'] = result['prob_broken'] >= self.broken_threshold

        return result

    def add_cathode_points(self, mask):
        """
        Combine predicted cathode mask with current state
        :param mask: predicted mask from Segmentation Neural Network
        """

        self.cathode_points = self.find_cathodes(mask)

    def add_anode_points(self, mask):
        """
        Combine predicted anode mask with current state
        :param mask: predicted mask from Segmentation Neural Network
        """

        self.anode_points = self.find_anodes(mask)

    def find_anodes(self, mask):
        result = []

        model = DBSCAN(eps=4, min_samples=15)
        x, y = np.where(mask >= self.anode_threshold)
        XY = np.vstack([x, y]).T
        if len(XY) > 0:
            pred = model.fit_predict(XY)
            num_clusters = pred.max() + 1
            for i in range(num_clusters):
                xy_cluster = XY[np.where(pred == i)[0]]
                result.append([int(xy_cluster[:, 0].mean()), int(xy_cluster[:, 1].mean()), xy_cluster.shape[0]])

        result = sorted(result, key=lambda x: x[1])
        return result

    def find_cathodes(self, mask):
        result = []

        model = DBSCAN(eps=4, min_samples=15)
        x, y = np.where(mask >= self.cathode_threshold)
        XY = np.vstack([x, y]).T
        if len(XY) > 0:
            pred = model.fit_predict(XY)
            num_clusters = pred.max() + 1
            for i in range(num_clusters):
                xy_cluster = XY[np.where(pred == i)[0]]
                result.append([int(xy_cluster[:, 0].mean()), int(xy_cluster[:, 1].mean()), xy_cluster.shape[0]])

        result = sorted(result, key=lambda x: x[1])
        return result

    def draw_points(self, points, color):
        """
        Draw points with color on image for drawing
        :param points: points to draw
        :param color: color to use
        """
        for p in points:
            if len(p) == 3:
                xx, yy, w = p
            else:
                xx, yy = p
            cv2.circle(self.img_draw, (int(yy), int(xx)), 5, color)

    def draw_text(self, data):
        for p, text in data:
            cv2.putText(self.img_draw, text, (p[1] - 30, p[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)

    def find_pairs(self):
        """
        Crop cathode/anode pairs from source image
        For each cathode crop Left and Right pair. Exception: 2-nd cathode(Left pair), penultimate cathode(Right pair)
        :return: List of crops
        """

        assert len(self.cathode_points) == self.cathode_count
        assert len(self.anode_points) == self.cathode_count + 1

        res = []

        for i, (x, y, w) in enumerate(self.cathode_points):
            for side in ['L', 'R']:
                if (i != 1 and side == 'L') or (i != self.cathode_count - 2 and side == "R"):
                    # noinspection PyDictCreation
                    item = dict()
                    item['side'] = side
                    item['cathode'] = len(self.cathode_points) - i
                    item['anode'] = item['cathode'] + (1 if side == "L" else 0)
                    item['cathode_point'] = self.cathode_points[self.cathode_count-item['cathode']]
                    item['anode_point'] = self.anode_points[self.cathode_count-item['anode']+1]

                    item['dist'] = item['anode_point'][0] - item['cathode_point'][0]
                    item['prob_broken'] = self.prob_broken(item['dist'])
                    item['cathode_w'] = float(w)
                    item['anode_w'] = float(item['anode_point'][2])

                    item['cathode_point'] = item['cathode_point'][:2]
                    item['anode_point'] = item['anode_point'][:2]
                    res.append(item)
        return res

    def prob_broken(self, dist: float):
        intervals = [
            (-10**6, 0, 1),
            (0, 2, 0.95),
            (2, 4, 0.88),
            (4, 6, 0.8),
            (6, 7, 0.65),
            (7, 8, 0.58),
            (8, 9, 0.5),
            (9, 10, 0.35),
            (10, 11, 0.2),
            (11, 13, 0.1),
            (13, 15, 0.05),
            (15, 20, 0.02),
            (20, 56, 0),
            (56, 61, 0.02),
            (61, 63, 0.05),
            (63, 65, 0.1),
            (65, 66, 0.2),
            (66, 67, 0.35),
            (67, 68, 0.5),
            (68, 69, 0.58),
            (69, 70, 0.65),
            (70, 72, 0.8),
            (72, 74, 0.88),
            (74, 76, 0.95),
            (76, 10**6, 0.95),
        ]

        for l, r, p in intervals:
            if l <= dist < r:
                return p
        raise Exception('wrong dist')