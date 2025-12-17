import numpy as np
from scipy.interpolate import interp1d

class EdgeComparator:
    #Vergleicht zwei Puzzle-Kanten.

    def __init__(self, edge_a, edge_b, num_points=100):
        self.edge_a = np.array(edge_a)
        self.edge_b = np.array(edge_b)
        self.num_points = num_points

    def _normalize_geometry(self, edge: np.ndarray) -> np.ndarray:
        edge = np.asanyarray(edge)
        if len(edge) < 2:
            return edge
            
        #Translation für optimalen Vergleich
        edge = edge - edge[0]
        
        #Rotation
        end_point = edge[-1]
        angle = -np.arctan2(end_point[1], end_point[0])
        c, s = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[c, -s], [s, c]])
        edge = np.dot(edge, rot_matrix.T)
        
        # Skalierung
        x_dist = edge[-1, 0]
        if x_dist > 0:
            edge = edge / x_dist
            
        return edge

    def _resample_edge(self, edge: np.ndarray) -> np.ndarray:
        #Interpoliert die Kante auf eine feste Punktanzahl entlang der Kurve.
        edge = np.asanyarray(edge)
        if len(edge) < 2: 
            return edge
        
        diffs = np.linalg.norm(np.diff(edge, axis=0), axis=1)
        dists = np.concatenate([[0], np.cumsum(diffs)])
        total_len = dists[-1]
        
        if total_len == 0: 
            return edge

        fx = interp1d(dists, edge[:, 0], kind='linear')
        fy = interp1d(dists, edge[:, 1], kind='linear')
        
        new_dists = np.linspace(0, total_len, self.num_points)
        new_edge = np.stack([fx(new_dists), fy(new_dists)], axis=1)
        
        return new_edge

    def get_edge_type(self, edge_norm: np.ndarray):
        #Klassifiziert die Kante: tab, hole, flat

        threshold = 0.12

        ys = edge_norm[:, 1]
        max_y = np.max(ys)
        min_y = np.min(ys)
        
        if max_y > threshold:
            return "tab"
        elif min_y < -threshold:
            return "hole"
        else:
            return "flat"

    def compare(self) -> float:
        #Vergleich zweier Kanten.

        #Normalisieren
        A_norm = self._normalize_geometry(self.edge_a)
        B_norm = self._normalize_geometry(self.edge_b)
        
        A = self._resample_edge(A_norm)
        B = self._resample_edge(B_norm)
        
        #Typ-Filter
        type_a = self.get_edge_type(A)
        type_b = self.get_edge_type(B)
        
        if type_a == "flat" or type_b == "flat":
            return 99.0
            
        if type_a == type_b:
            return 98.0
            
        #Geometrischer Vergleich
        B_inv = B.copy()
        B_inv[:, 1] *= -1 
        
        # Teste beide Richtungen
        diff_fwd = np.sqrt(np.mean(np.sum((A - B_inv)**2, axis=1)))
        
        B_inv_rev = B_inv[::-1].copy()
        B_inv_rev[:, 0] = 1.0 - B_inv_rev[:, 0]
        diff_rev = np.sqrt(np.mean(np.sum((A - B_inv_rev)**2, axis=1)))
        
        shape_score = min(diff_fwd, diff_rev)
        
        #Höhen-Check
        height_a = np.max(np.abs(A[:, 1]))
        height_b = np.max(np.abs(B[:, 1]))
        height_penalty = abs(height_a - height_b)
        
        return shape_score + (height_penalty * 0.5)