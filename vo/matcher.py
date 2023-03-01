import cv2

FLANN_INDEX_LSH = 6


class FlannMatcher(object):
    def __init__(
        self,
        index_params=dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1),
        search_params=dict(checks=50)
    ):
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    def match(self, descs1, descs2):
        matches = self.flann.knnMatch(descs1, descs2, k=2)
        good_matches = []
        for m in matches:
            if len(m) < 2:
                continue
            if m[0].distance < 0.7 * m[1].distance:
                good_matches.append(m[0])
        return good_matches
