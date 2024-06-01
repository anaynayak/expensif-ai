from typing import List
from parser.vision import Observation


def cluster(observations: List[Observation]):
    observations = [observation.expand_till_left() for observation in observations]
    clusters: List[Observation] = []
    for observation in observations:
        overlaps = False
        new_clusters = []
        for other_observation in clusters:
            if observation.overlaps(other_observation) and not overlaps:
                new_clusters.append(observation.merge(other_observation))
                overlaps = True
            else:
                new_clusters.append(other_observation)
        clusters = new_clusters
        if not overlaps:
            clusters.append(observation)
    return clusters
