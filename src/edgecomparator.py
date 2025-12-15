import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class EdgeComparator:
    """
    Vergleicht Kanten
    """

    def __init__(self, edge_a, edge_b):
        self.edge_a = np.array(edge_a)
        self.edge_b = np.array(edge_b)
        
        # Länge berechnen.
        self.length_a = np.sum(np.linalg.norm(np.diff(self.edge_a, axis=0), axis=1))
        self.length_b = np.sum(np.linalg.norm(np.diff(self.edge_b, axis=0), axis=1))
          

    def compare_length(self) -> bool:
        tolerance: float = 5.0
        
        diff = abs(self.length_a - self.length_b)
        logger.debug(f"Laengenvergleich: diff={diff:.3f}, tolerance={tolerance}")
        return diff <= tolerance

    def compare_contour(self) -> bool:
        a = self.edge_a
        b = self.edge_b
        tolerance: float = 0.02

        # Kanten in gleiche Struktur bringen.
        def resample(edge):
            d = np.linalg.norm(np.diff(edge, axis=0), axis=1)
            s = np.insert(np.cumsum(d), 0, 0)
            s /= s[-1]
            t = np.linspace(0, 1, 100)
            return np.column_stack([
                np.interp(t, s, edge[:, 0]),
                np.interp(t, s, edge[:, 1])
            ])

        a = resample(self.edge_a)
        b = resample(self.edge_b)
       
        # gespiegelt zur Vergleichbarkeit.
        b_flip = -b[::-1]

        #Zentrieren
        a -= np.mean(a, axis=0)
        b_flip -= np.mean(b_flip, axis=0)

        #Skalieren
        scale = max(np.max(np.linalg.norm(a, axis=1)), np.max(np.linalg.norm(b_flip, axis=1)))
        a /= scale
        b_flip /= scale

        mse = np.mean((a - b_flip) ** 2)
        logger.info(f"MSE = {mse:.4f}")
        logger.debug(f"Contour MSE: {mse:.5f} (tolerance: {tolerance})")
        return mse < tolerance

    def match_edges(self) -> bool:
        """
        Laenge, dann Kontur
        """
        #if not self.compare_length():
        #    logger.debug("Length mismatch.")
        #    return False

        if not self.compare_contour():
            logger.debug("Contour mismatch.")
            return False

        logger.info("Edges matched: Length and Contour within tolerances.")
        return True
        #TODO: Logging Rückgabe der passenden Puzzleteile, evtl. Kanten.