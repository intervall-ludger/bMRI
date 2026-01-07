from __future__ import annotations

import sys
from typing import Callable, Sequence

import numpy as np
from PyQt5.QtWidgets import QApplication

from src.Visualization.image_viewer import ImageViewer


def show(
    dicom: np.ndarray,
    fit_maps: np.ndarray,
    fit: Callable[..., np.ndarray],
    time_points: Sequence[float],
) -> None:
    # Create an instance of the ImageViewer class

    app = QApplication(sys.argv)

    viewer = ImageViewer(dicom, fit_maps, fit, time_points, 1)
    viewer.show()

    # Run the PyQt5 application
    sys.exit(app.exec_())
