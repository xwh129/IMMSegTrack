# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from .kalman_filter import chi2inv95

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.yolo.utils.checks import check_requirements

    check_requirements('lap>=0.4')  # install
    import lap


def merge_matches(m1, m2, shape):
    """Merge two sets of matches and return matched and unmatched indices."""
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - {i for i, j in match})
    unmatched_Q = tuple(set(range(Q)) - {j for i, j in match})

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    """_indices_to_matches: Return matched and unmatched indices given a cost matrix, indices, and a threshold."""
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh, use_lap=True):
    """Linear assignment implementations with scipy and lap.lapjv."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # Scipy linear sum assignment is NOT working correctly, DO NOT USE
        y, x = scipy.optimize.linear_sum_assignment(cost_matrix)  # row y, col x
        matches = np.asarray([[i, x] for i, x in enumerate(x) if cost_matrix[i, x] <= thresh])
        unmatched = np.ones(cost_matrix.shape)
        for i, xi in matches:
            unmatched[i, xi] = 0.0
        unmatched_a = np.where(unmatched.all(1))[0]
        unmatched_b = np.where(unmatched.all(0))[0]

    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious

    ious = bbox_ious(np.ascontiguousarray(atlbrs, dtype=np.float32), np.ascontiguousarray(btlbrs, dtype=np.float32))
    return ious

def ious_occlusion(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    ious = bbox_ious_occlusion(np.ascontiguousarray(atlbrs, dtype=np.float32), np.ascontiguousarray(btlbrs, dtype=np.float32))
    return ious

def eucli(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious

    ious = bbox_dis(np.ascontiguousarray(atlbrs, dtype=np.float32), np.ascontiguousarray(btlbrs, dtype=np.float32))
    return ious

def ious_mask(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    
    from shapely.geometry import Polygon
    for i in range(len(atlbrs)):
        for j in range(len(btlbrs)):
            poly1=Polygon(atlbrs[i].reshape(-1,2))
            poly2=Polygon(btlbrs[j].reshape(-1,2))

            try:
                intersection_area = poly1.intersection(poly2).area
                union_area = poly1.union(poly2).area
            except:
                poly1 = poly1.buffer(0.01)
                poly2 = poly2.buffer(0.01)
                intersection_area = poly1.intersection(poly2).area
                union_area = poly1.union(poly2).area
            # 计算交并比
            iou = intersection_area / union_area
            ious[i][j]=iou
    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
            or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
        print(atlbrs,btlbrs)
        exit()
    else:
        # atlbrs = [track.tlbr for track in atracks]
        # btlbrs = [track.tlbr for track in btracks]
        amasks = [track.mask for track in atracks]
        bmasks = [track.mask for track in btracks]
    _ious = ious_mask(amasks, bmasks)
    # _ious = ious(atlbrs, btlbrs)
    return 1 - _ious  # cost matrix

def iou_distance_ori(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
            or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
        print(atlbrs,btlbrs)
        exit()
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    return 1 - _ious  # cost matrix

def eucli_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
            or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
        print(atlbrs,btlbrs)
        exit()
    else:
        amasks = [track.tlbr for track in atracks]
        bmasks = [track.tlbr for track in btracks]
        aocclusion=np.array([track.occlusion_iou for track in atracks])
    _ious = eucli(amasks, bmasks)
    _ious[aocclusion>0.8]=10000
    return _ious  # cost matrix


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def eucli_cosine_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
            or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
        print(atlbrs,btlbrs)
        exit()
    else:
        amasks = [track.tlbr for track in atracks]
        bmasks = [track.tlbr for track in btracks]

        # afeatures = [track.feature_history for track in atracks]
        # bfeatures = [track.feature_history for track in btracks]
        aocclusion=np.array([track.occlusion_iou for track in atracks])

    _ious = eucli(amasks, bmasks)
    _ious[aocclusion>0.8]=10000 
    # _ious[_cosine>0.025]=10000
    return _ious  # cost matrix



def iou_occlusion(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
            or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
        print(atlbrs,btlbrs)
        exit()
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        # amasks = [track.mask for track in atracks]
        # bmasks = [track.mask for track in btracks]
    # _ious = ious_mask(amasks, bmasks)
    _ious = ious_occlusion(atlbrs, btlbrs)
    return 1 - _ious  # cost matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
            or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    return 1 - _ious  # cost matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """Apply gating to the cost matrix based on predicted tracks and detected objects."""
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """Fuse motion between tracks and detections with gating and Kalman filtering."""
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    """Fuses ReID and IoU similarity matrices to yield a cost matrix for object tracking."""
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    # det_scores = np.array([det.score for det in detections])
    # det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    return 1 - fuse_sim  # fuse cost


def fuse_score(cost_matrix, detections):
    """Fuses cost matrix with detection scores to produce a single similarity matrix."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost


def bbox_ious(box1, box2, eps=1e-7):
    """
    Calculate the Intersection over Union (IoU) between pairs of bounding boxes.

    Args:
        box1 (np.array): A numpy array of shape (n, 4) representing 'n' bounding boxes.
                         Each row is in the format (x1, y1, x2, y2).
        box2 (np.array): A numpy array of shape (m, 4) representing 'm' bounding boxes.
                         Each row is in the format (x1, y1, x2, y2).
        eps (float, optional): A small constant to prevent division by zero. Defaults to 1e-7.

    Returns:
        (np.array): A numpy array of shape (n, m) representing the IoU scores for each pair
                    of bounding boxes from box1 and box2.

    Note:
        The bounding box coordinates are expected to be in the format (x1, y1, x2, y2).
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # box2 area
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (box2_area + box1_area[:, None] - inter_area + eps)

def bbox_ious_occlusion(box1, box2, eps=1e-7):
    """
    Calculate the Intersection over Union (IoU) between pairs of bounding boxes.

    Args:
        box1 (np.array): A numpy array of shape (n, 4) representing 'n' bounding boxes.
                         Each row is in the format (x1, y1, x2, y2).
        box2 (np.array): A numpy array of shape (m, 4) representing 'm' bounding boxes.
                         Each row is in the format (x1, y1, x2, y2).
        eps (float, optional): A small constant to prevent division by zero. Defaults to 1e-7.

    Returns:
        (np.array): A numpy array of shape (n, m) representing the IoU scores for each pair
                    of bounding boxes from box1 and box2.

    Note:
        The bounding box coordinates are expected to be in the format (x1, y1, x2, y2).
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # box2 area
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    return inter_area / (box1_area[:, None] + eps)

def bbox_dis(box1, box2, eps=1e-7):
    """
    Calculate the Intersection over Union (IoU) between pairs of bounding boxes.

    Args:
        box1 (np.array): A numpy array of shape (n, 4) representing 'n' bounding boxes.
                         Each row is in the format (x1, y1, x2, y2).
        box2 (np.array): A numpy array of shape (m, 4) representing 'm' bounding boxes.
                         Each row is in the format (x1, y1, x2, y2).
        eps (float, optional): A small constant to prevent division by zero. Defaults to 1e-7.

    Returns:
        (np.array): A numpy array of shape (n, m) representing the IoU scores for each pair
                    of bounding boxes from box1 and box2.

    Note:
        The bounding box coordinates are expected to be in the format (x1, y1, x2, y2).
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    b1_center=(b1_x1+b1_x2)/2
    b2_center=(b2_x1+b2_x2)/2
    dis=np.sqrt((b1_center[:,None]-b2_center)*(b1_center[:,None]-b2_center))
    return dis
