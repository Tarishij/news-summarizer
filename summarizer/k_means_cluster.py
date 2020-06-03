import numpy as np
from sklearn.cluster import KMeans
from typing import List

def __find_closest_args(features,centroids: np.ndarray):
        #Find the closest arguments to centroid

        centroid_min = 1e10
        cur_arg = -1
        args = {}
        used_idx = []

        for j, centroid in enumerate(centroids):

            for i, feature in enumerate(features):
                value = np.linalg.norm(feature - centroid)

                if value < centroid_min and i not in used_idx:
                    cur_arg = i
                    centroid_min = value

            used_idx.append(cur_arg)
            args[j] = cur_arg
            centroid_min = 1e10
            cur_arg = -1

        return args

def cluster(features,algorithm,random_state, summary_length=20) -> List[int]:
        #Clusters sentences based on the ratio
        
        #print(summary_length)
        #print(len(features))
        if summary_length*3<len(features):
            k=summary_length*3
        else:
            k=len(features)
        
        model = KMeans(n_clusters=k, random_state=random_state).fit(features)
        
        centroids = model.cluster_centers_

        cluster_args = __find_closest_args(features,centroids)
        
        sorted_values = sorted(cluster_args.values())
        #Sentences index that qualify for summary
        return sorted_values


