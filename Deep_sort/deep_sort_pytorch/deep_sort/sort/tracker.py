from __future__ import absolute_import
import numpy as np

# *.py files
from .tracker_ import kalman_filter
from .tracker_ import linear_assignmnet
from .tracker_ import iou_matching
from .tracker_.track import Track

a = 1
class Tracker:

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = a

    def predict(self):

        for track in self.tracks:
            track.predict(self.kf)


    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update( self.kf, detections[detection_idx])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            new_id = self._initiate_track(detections[detection_idx])
            self._initiate_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []

        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):

            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignmnet.gate_cost_matrix(self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)

            return cost_matrix

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignmnet.matching_cascade(gated_metric, self.metric.matching_threshold,
                                                                                                self.max_age, self.tracks, detections, confirmed_tracks)

        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1 ]
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignmnet.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance,
                                                                                                 self.tracks, detections, iou_track_candidates,
                                                                                                 unmatched_detections)
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        cls = detection.clses
        score = detection.confidence
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, cls, score, detection.feature))
        self._next_id += 1
        a = self._next_id
        return (self._next_id - 1)


