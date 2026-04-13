import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from numba import njit
from scipy import ndimage

from src.Utilitis import load_nii
from src.Utilitis.utils import get_function_parameter


def get_crop_window(heat_map: np.ndarray, padding: int = 10) -> tuple[int, int]:
    """Calculate crop window from heatmap non-zero region."""
    not_nan_indices = np.where(heat_map[0] != 0)
    if len(not_nan_indices[0]) == 0:
        return (0, heat_map.shape[1])
    slice_x = (
        max(0, not_nan_indices[0].min() - padding),
        min(heat_map.shape[1], not_nan_indices[0].max() + padding),
    )
    slice_y = (
        max(0, not_nan_indices[1].min() - padding),
        min(heat_map.shape[2], not_nan_indices[1].max() + padding),
    )
    return (max(slice_x[0], slice_y[0]), max(slice_x[1], slice_y[1]))


def crop_to_heatmap(
    dicom_image: np.ndarray, heat_map: np.ndarray, padding: int = 10
) -> tuple:
    """
    Crops the given DICOM image and heatmap to the non-NaN region of the heatmap.

    :param dicom_image: 3D array representing the DICOM image.
    :param heat_map: 3D array representing the heatmap, which can include NaN values.
    :param padding: Additional padding to be added around the non-NaN region of the heatmap.
    :return: A tuple containing the cropped DICOM image and heatmap.
    """
    window = get_crop_window(heat_map, padding)
    dicom_image_cropped = dicom_image[
        :, window[0] : window[1], window[0] : window[1], :
    ]
    heat_map_cropped = heat_map[:, window[0] : window[1], window[0] : window[1], :]

    return dicom_image_cropped, heat_map_cropped


def calc_scaling_factor(dicom_shape: tuple[int, int, int]) -> int:
    """
    Calculate scaling factor based on DICOM shape.

    :param dicom_shape: Shape of the DICOM data.
    :return: Scaling factor.
    """
    return 500 // max(dicom_shape[1], dicom_shape[2])


class ImageViewer(QMainWindow):
    """
    ImageViewer class to visualize DICOM data and fitted maps.
    """

    def __init__(self):
        super().__init__()

    def start(
        self,
        dicom: np.ndarray | Path,
        fit_maps: np.ndarray | list | Path,
        fit_function: callable,
        time_points: list[int] | np.ndarray,
        c_int: int | None = None,
        alpha: float = 0.35,
        normalize: bool = True,
        auto_cut: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        mask_file: Path | None = None,
    ):
        """
        Initialize the ImageViewer.

        :param dicom: DICOM data array.
        :param fit_maps: Array of fitted maps.
        :param fit_function: Fitting function.
        :param time_points: List of time points.
        :param c_int: Color intensity index, optional.
        :param alpha: Alpha value for overlay, optional.
        :param normalize: Flag to normalize data, optional.
        """
        if isinstance(dicom, Path):
            dicom = load_nii(dicom).array
        if isinstance(fit_maps, Path):
            fit_maps = load_nii(fit_maps).array
            fit_maps[fit_maps == -1] = 0

        # Load mask before cropping so we can crop it with the same window
        mask_array = None
        if mask_file is not None and Path(mask_file).exists():
            mask_array = load_nii(mask_file).array

        if auto_cut:
            window = get_crop_window(fit_maps, 50)
            dicom = dicom[:, window[0]:window[1], window[0]:window[1], :]
            fit_maps = fit_maps[:, window[0]:window[1], window[0]:window[1], :]
            if mask_array is not None:
                mask_array = mask_array[window[0]:window[1], window[0]:window[1], :]

        if isinstance(time_points, np.ndarray):
            time_points = list(time_points)

        self.echo_time = 0
        self.time_points = time_points
        self.dicom = dicom
        self.alpha = alpha
        self.norm = normalize
        self.fit_maps = np.array(fit_maps)
        self.fit_function = fit_function
        self.vmin_override = vmin
        self.vmax_override = vmax
        self.parameter_names = list(get_function_parameter(self.fit_function))
        if not self.parameter_names:
            self.parameter_names = [f"Param {idx+1}" for idx in range(self.fit_maps.shape[0])]
        self.current_param_index = c_int if c_int is not None else 1 if len(self.parameter_names) > 1 else 0
        self.current_param_index = min(max(self.current_param_index, 0), len(self.parameter_names) - 1)
        self.color_map = self.fit_maps[self.current_param_index]
        self.colorbar_range = self._compute_color_range()
        self.base_scaling_factor = calc_scaling_factor(dicom.shape)
        self.zoom_level = 1.0
        self.scaling_factor = self.base_scaling_factor
        self.current_slice = self._find_best_initial_slice()
        self.current_params = self.fit_maps[:, :, :, self.current_slice]

        # Mask data for ROI statistics (already cropped above if auto_cut)
        self.mask_file = mask_file
        self.mask_data = mask_array

        # Create a label to display the image
        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid #222")
        self.image_label.mousePressEvent = self.update_fit_function
        width, height = (
            dicom.shape[1] * self.scaling_factor,
            dicom.shape[2] * self.scaling_factor,
        )
        self.image_label.setFixedSize(width, height)

        # Display the first slice
        self.display_slice()

        # Container for fit plot
        self.plot_container = QWidget(self)
        self.plot_container.setMinimumWidth(320)

        # Set up the axes, but don't plot any data initially
        self.x_fit = np.linspace(0, self.time_points[-1], 1000)

        # Set the layout of the FitFunctionWidget
        layout = QHBoxLayout()
        self.plot_container.setLayout(layout)

        self.init_fit_function()

        controls_widget = self._create_controls()

        # Set the layout of the ImageViewer
        main_layout = QHBoxLayout()
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.plot_container)
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self._apply_global_style()
        self.update_colorbar()

    def _apply_global_style(self) -> None:
        """Apply a subtle dark theme to the viewer window."""
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0b0c10;
            }
            QWidget#sectionFrame {
                background-color: #12141c;
                border: 1px solid #1f2230;
                border-radius: 6px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QLabel.sectionTitle {
                font-weight: 600;
                letter-spacing: 0.5px;
                color: #7cd5ff;
            }
            QComboBox {
                background-color: #1b1e29;
                border: 1px solid #2a2f3f;
                padding: 4px;
                border-radius: 4px;
            }
            QSlider::groove:horizontal, QSlider::groove:vertical {
                border: 1px solid #2a2f3f;
                background: #181a22;
            }
            QSlider::handle:horizontal, QSlider::handle:vertical {
                background: #7cd5ff;
                border: 1px solid #22283a;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """
        )

    def change_slice(self, slice_num: int):
        """
        Change displayed slice.

        :param slice_num: Slice number to display.
        """
        self.current_slice = slice_num
        self.current_params = self.fit_maps[:, :, :, self.current_slice]
        if hasattr(self, "slice_value_label"):
            self.slice_value_label.setText(
                f"{slice_num + 1} / {self.dicom.shape[-1]}"
            )
        self.display_slice()

    def _on_slice_slider_changed(self, value: int) -> None:
        """Handle slider changes and keep the label in sync."""
        self.change_slice(value)

    def display_slice(self):
        """
        Display the current slice.
        """
        # Get the size of the widget
        size = self.image_label.size()

        # Get the image slice and normalize it to the range [0, 1]
        image = self.dicom[self.echo_time, :, :, self.current_slice]
        image = (image - image.min()) / (image.max() - image.min())

        # Zoom the image by a factor of 5
        image_zoomed = ndimage.zoom(
            image, (self.scaling_factor, self.scaling_factor), order=0, mode="nearest"
        )

        # Convert the zoomed image to an RGB image
        image_zoomed_rgb = np.dstack((image_zoomed, image_zoomed, image_zoomed))

        # Normalize the color map to the range [0, 1]
        if self.color_map is not None:
            color_map = self.color_map[:, :, self.current_slice]
            vmin, vmax = self.colorbar_range
            denom = vmax - vmin if vmax != vmin else 1.0
            color_map_norm = np.clip((color_map - vmin) / denom, 0, 1)
            color_map_norm = np.nan_to_num(color_map_norm, nan=0.0)

            color_map_zoomed = ndimage.zoom(
                color_map_norm,
                (self.scaling_factor, self.scaling_factor),
                order=0,
                mode="nearest",
            )
            jet_cmap = get_cmap("jet")
            color_map_zoomed_rgb = jet_cmap(color_map_zoomed)[..., :3]

            overlay_mask = color_map_zoomed > 0
            if np.any(overlay_mask):
                base = image_zoomed_rgb.astype(float)
                base[overlay_mask] = (
                    (1 - self.alpha) * base[overlay_mask]
                    + self.alpha * color_map_zoomed_rgb[overlay_mask]
                )
                image_zoomed_rgb = base

        image_zoomed_rgb = np.clip(image_zoomed_rgb * 255, 0, 255).astype("uint8")
        # Convert the zoomed RGB image to a QImage and create a QPixmap from it
        qimage = QImage(
            image_zoomed_rgb,
            image_zoomed_rgb.shape[1],
            image_zoomed_rgb.shape[0],
            image_zoomed_rgb.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)

        # Set the pixmap as the background image of the label
        self.image_label.setPixmap(pixmap)

    def _create_controls(self) -> QWidget:
        """Create the control sidebar with sliders and metadata panels."""
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(14)

        slice_section = self._make_section("Slice Navigation")
        slice_layout = slice_section.layout()
        self.slice_value_label = QLabel(f"{self.current_slice + 1} / {self.dicom.shape[-1]}")
        slice_layout.addWidget(self.slice_value_label)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.dicom.shape[-1] - 1)
        self.slice_slider.setValue(self.current_slice)
        self.slice_slider.valueChanged.connect(self._on_slice_slider_changed)
        self.slice_slider.setMinimumHeight(28)
        self.slice_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        slice_layout.addWidget(self.slice_slider)
        controls_layout.addWidget(slice_section)

        parameter_section = self._make_section("Parameter Overlay")
        param_layout = parameter_section.layout()
        self.parameter_combo = QComboBox()
        self.parameter_combo.addItems([name.upper() for name in self.parameter_names])
        self.parameter_combo.setCurrentIndex(self.current_param_index)
        self.parameter_combo.currentIndexChanged.connect(self.on_parameter_changed)
        param_layout.addWidget(self.parameter_combo)
        self.parameter_summary = QLabel(
            f"Highlighting: {self.parameter_names[self.current_param_index].upper()}"
        )
        self.parameter_summary.setStyleSheet("color: #a0a7c2; font-size: 12px;")
        param_layout.addWidget(self.parameter_summary)
        controls_layout.addWidget(parameter_section)

        alpha_section = self._make_section("Overlay Opacity")
        alpha_layout = alpha_section.layout()
        self.alpha_value_label = QLabel(f"{self.alpha:.2f}")
        alpha_layout.addWidget(self.alpha_value_label)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(self.alpha * 100))
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        self.alpha_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        alpha_layout.addWidget(self.alpha_slider)
        controls_layout.addWidget(alpha_section)

        zoom_section = self._make_section("Zoom")
        zoom_layout = zoom_section.layout()
        self.zoom_value_label = QLabel(f"{self.zoom_level:.1f}x")
        zoom_layout.addWidget(self.zoom_value_label)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 50)
        self.zoom_slider.setValue(10)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        self.zoom_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_hint = QLabel("Or use mouse wheel on image")
        zoom_hint.setStyleSheet("color: #666; font-size: 10px;")
        zoom_layout.addWidget(zoom_hint)
        controls_layout.addWidget(zoom_section)

        color_section = self._make_section("Color Scale")
        color_layout = color_section.layout()
        self.colorbar_fig = Figure(figsize=(1.8, 3.0))
        self.colorbar_fig.patch.set_facecolor("#ffffff")
        self.colorbar_canvas = FigureCanvas(self.colorbar_fig)
        self.colorbar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.colorbar_ax = self.colorbar_fig.add_subplot(111)
        self.colorbar_ax.set_facecolor("#111111")
        color_layout.addWidget(self.colorbar_canvas)
        spin_layout = QHBoxLayout()
        self.vmin_spin = self._make_spinbox("Min")
        self.vmax_spin = self._make_spinbox("Max")
        self.vmin_spin.valueChanged.connect(self.on_color_spin_changed)
        self.vmax_spin.valueChanged.connect(self.on_color_spin_changed)
        spin_layout.addWidget(self.vmin_spin)
        spin_layout.addWidget(self.vmax_spin)
        color_layout.addLayout(spin_layout)
        self.reset_color_btn = QPushButton("Auto-scale")
        self.reset_color_btn.clicked.connect(self.on_color_reset)
        self.reset_color_btn.setStyleSheet(
            "background-color:#1f2435; border:1px solid #2e3550; padding:6px; border-radius:4px; color:#e0e6ff;"
        )
        color_layout.addWidget(self.reset_color_btn)
        self._sync_color_spins()
        controls_layout.addWidget(color_section)

        self.info_panel = InfoPanel(self.parameter_names)
        controls_layout.addWidget(self.info_panel)

        self.roi_summary = ROISummaryPanel()
        self.roi_summary.update_summary(self.fit_maps, self.current_param_index, self.mask_data)
        controls_layout.addWidget(self.roi_summary)

        controls_layout.addStretch(1)

        widget = QWidget()
        widget.setLayout(controls_layout)
        return widget

    def _make_section(self, title: str) -> QFrame:
        """Create a styled section container."""
        frame = QFrame()
        frame.setObjectName("sectionFrame")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        label = QLabel(title.upper())
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        return frame

    def _make_spinbox(self, placeholder: str) -> QDoubleSpinBox:
        """Create a compact spinbox for color scale input."""
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-1e6, 1e6)
        spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        spin.setMinimumWidth(70)
        spin.setSpecialValueText(placeholder)
        spin.setStyleSheet(
            "background-color:#1b1e29; border:1px solid #2a2f3f; padding:4px; border-radius:4px; color:#e0e0e0;"
        )
        return spin

    def _find_best_initial_slice(self) -> int:
        """Find the slice with the most non-zero pixels in the color map."""
        if self.color_map is None:
            return 0
        pixel_counts = []
        for s in range(self.color_map.shape[2]):
            slice_data = self.color_map[:, :, s]
            count = np.sum((slice_data > 0) & np.isfinite(slice_data))
            pixel_counts.append(count)
        if not pixel_counts or max(pixel_counts) == 0:
            return 0
        return int(np.argmax(pixel_counts))

    def _compute_color_range(self) -> tuple[float, float]:
        """Calculate reasonable color limits for the overlay."""
        if self.vmin_override is not None or self.vmax_override is not None:
            finite = self.color_map[np.isfinite(self.color_map)] if self.color_map is not None else np.array([0.0, 1.0])
            vmin = float(self.vmin_override) if self.vmin_override is not None else float(np.nanmin(finite))
            vmax = float(self.vmax_override) if self.vmax_override is not None else float(np.nanmax(finite))
            if vmin >= vmax:
                vmax = vmin + 1.0
            return vmin, vmax
        if self.color_map is None:
            return (0.0, 1.0)
        finite = self.color_map[np.isfinite(self.color_map)]
        if finite.size == 0:
            return (0.0, 1.0)
        vmin = float(np.nanpercentile(finite, 1))
        vmax = float(np.nanpercentile(finite, 99))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    def update_colorbar(self) -> None:
        """Update the colorbar to match the current parameter range."""
        if self.color_map is None:
            return

        vmin, vmax = self._compute_color_range()
        self.colorbar_range = (vmin, vmax)
        self._sync_color_spins()

        gradient = np.linspace(0, 1, 256).reshape(256, 1)
        self.colorbar_ax.clear()
        # Leave a larger left margin (white space) while keeping a compact bar
        self.colorbar_ax.set_position([0.35, 0.05, 0.4, 0.9])
        self.colorbar_ax.imshow(gradient, aspect="auto", cmap="jet", origin="lower")
        self.colorbar_ax.set_xticks([])
        ticks = [0, 128, 255]
        tick_labels = [f"{vmin:.2f}", f"{(vmin + vmax)/2:.2f}", f"{vmax:.2f}"]
        self.colorbar_ax.set_yticks(ticks)
        self.colorbar_ax.set_yticklabels(tick_labels, color="#dcdcdc")
        self.colorbar_ax.tick_params(axis="y", colors="#dcdcdc", labelsize=9)
        self.colorbar_ax.set_title(
            self.parameter_names[self.current_param_index].upper(),
            fontsize=10,
            color="#f8f8f8",
        )
        self.colorbar_canvas.draw()
        if hasattr(self, "parameter_summary"):
            self.parameter_summary.setText(
                f"Highlighting: {self.parameter_names[self.current_param_index].upper()}"
            )

    def on_alpha_changed(self, value: int) -> None:
        """Handle overlay opacity changes."""
        self.alpha = value / 100 or 0.01
        if hasattr(self, "alpha_value_label"):
            self.alpha_value_label.setText(f"{self.alpha:.2f}")
        self.display_slice()

    def on_parameter_changed(self, index: int) -> None:
        """Handle switching between overlay parameters."""
        if index < 0 or index >= len(self.parameter_names):
            return
        self.current_param_index = index
        self.color_map = self.fit_maps[index]
        # Reset overrides when switching parameters to avoid stale values
        self.vmin_override = None
        self.vmax_override = None
        if hasattr(self, "parameter_summary"):
            self.parameter_summary.setText(
                f"Highlighting: {self.parameter_names[index].upper()}"
            )
        if hasattr(self, "roi_summary"):
            self.roi_summary.update_summary(self.fit_maps, index, self.mask_data)
        self.update_colorbar()
        self.display_slice()

    def init_fit_function(self):
        """
        Initialize the fit function plot.
        """
        self.fit_function_widget = FitFunctionWidget(
            [np.NAN] * len(self.time_points),
            self.fit_function,
            [np.NAN] * len(self.current_params[:, 0, 0]),
            self.time_points,
            self,
        )
        self.plot_container.layout().addWidget(self.fit_function_widget)

    def update_fit_function(self, event):
        """
        Update the fit function plot based on an event.

        :param event: The triggered event.
        """
        x = event.pos().x() // self.scaling_factor
        y = event.pos().y() // self.scaling_factor
        try:
            pixel_params = self.current_params[:, y, x]
        except IndexError:
            return None
        raw_data = self.dicom[:, y, x, self.current_slice].astype("float64")
        if self.norm:
            raw_data /= raw_data.max()
        self.fit_function_widget.update_plot(pixel_params, raw_data)
        self.info_panel.update_info(x, y, self.current_slice, pixel_params)

    def on_color_spin_changed(self) -> None:
        """Apply manual color limits from spin boxes."""
        self.vmin_override = float(self.vmin_spin.value())
        self.vmax_override = float(self.vmax_spin.value())
        if self.vmin_override >= self.vmax_override:
            self.vmax_override = self.vmin_override + 1.0
        self.update_colorbar()
        self.display_slice()

    def _sync_color_spins(self) -> None:
        """Sync spin boxes with current colorbar range."""
        if not hasattr(self, "vmin_spin") or not hasattr(self, "vmax_spin"):
            return
        vmin, vmax = self.colorbar_range
        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)

    def on_color_reset(self) -> None:
        """Reset to automatic color scaling."""
        self.vmin_override = None
        self.vmax_override = None
        self.colorbar_range = self._compute_color_range()
        self._sync_color_spins()
        self.update_colorbar()
        self.display_slice()

    def on_zoom_changed(self, value: int) -> None:
        """Handle zoom slider changes."""
        self.zoom_level = value / 10.0
        self._apply_zoom()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if event.angleDelta().y() > 0:
            self.zoom_level = min(5.0, self.zoom_level + 0.2)
        else:
            self.zoom_level = max(1.0, self.zoom_level - 0.2)
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(self.zoom_level * 10))
        self.zoom_slider.blockSignals(False)
        self._apply_zoom()

    def _apply_zoom(self) -> None:
        """Apply current zoom level to the image."""
        self.scaling_factor = int(self.base_scaling_factor * self.zoom_level)
        if hasattr(self, "zoom_value_label"):
            self.zoom_value_label.setText(f"{self.zoom_level:.1f}x")
        width = self.dicom.shape[1] * self.scaling_factor
        height = self.dicom.shape[2] * self.scaling_factor
        self.image_label.setFixedSize(width, height)
        self.display_slice()


class FitFunctionWidget(QWidget):
    """
    Widget for displaying the fitting function.
    """

    def __init__(
        self,
        raw_data: list[float],
        fit_function: callable,
        params: list[float],
        time_points: list[int],
        parent: QWidget = None,
    ):
        """
        Initialize the FitFunctionWidget.

        :param raw_data: List of raw data points.
        :param fit_function: Fitting function.
        :param params: List of parameters for the fitting function.
        :param time_points: List of time points.
        :param parent: Parent widget, optional.
        """
        super().__init__(parent)
        self.y_raw = raw_data
        self.fit_function = fit_function
        self.params = params
        self.time_points = time_points

        # Add a plot area to the widget
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor("#0c0c0c")
        self.axes.tick_params(colors="#bbbbbb")
        for spine in self.axes.spines.values():
            spine.set_color("#555555")

        # Plot the raw data and fit function
        self.x_fit = np.linspace(0, self.time_points[-1], 1000)
        self.y_fit = self.fit_function(self.x_fit, *self.params)
        self.axes.plot(
            self.time_points,
            self.y_raw,
            "o",
            markersize=4,
            label="Raw data",
            color="#1f77b4",
        )
        self.axes.plot(
            self.x_fit,
            self.y_fit,
            "-",
            label="Fit",
            color="#ff7f0e",
        )
        self.axes.set_xlabel("Echo Time (ms)", color="#bbbbbb")
        self.axes.set_ylabel("Signal", color="#bbbbbb")
        self.axes.legend(loc="upper right", fontsize=8)
        self.axes.grid(color="#222222", linestyle="--", linewidth=0.5)

        # Set the layout of the widget
        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_plot(self, params=None, raw_data=None):
        """
        Update the plot with new parameters or raw data.

        :param params: List of new parameters, optional.
        :param raw_data: List of new raw data, optional.
        """
        if params is not None:
            self.params = params
        if raw_data is not None:
            self.y_raw = raw_data
        self.y_fit = self.fit_function(self.x_fit, *self.params)
        self.axes.clear()
        self.axes.set_facecolor("#0c0c0c")
        self.axes.tick_params(colors="#bbbbbb")
        for spine in self.axes.spines.values():
            spine.set_color("#555555")
        self.axes.plot(
            self.time_points,
            self.y_raw,
            "o",
            markersize=4,
            label="Raw data",
            color="#1f77b4",
        )
        self.axes.plot(
            self.x_fit,
            self.y_fit,
            "-",
            label="Fit",
            color="#ff7f0e",
        )
        self.axes.set_xlabel("Echo Time (ms)", color="#bbbbbb")
        self.axes.set_ylabel("Signal", color="#bbbbbb")
        self.axes.legend(loc="upper right", fontsize=8)
        self.axes.grid(color="#222222", linestyle="--", linewidth=0.5)

        self.canvas.draw()


class InfoPanel(QWidget):
    """Small panel that shows coordinates and parameter values."""

    def __init__(self, parameter_names: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.parameter_names = parameter_names
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.instructions = QLabel("Click inside the image to inspect a voxel.")
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("color: #888;")
        layout.addWidget(self.instructions)

        self.coord_label = QLabel("Voxel: –")
        self.coord_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.coord_label)

        self.table_label = QLabel("<i>No voxel selected</i>")
        self.table_label.setWordWrap(True)
        self.table_label.setTextFormat(Qt.RichText)
        self.table_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.table_label.setStyleSheet("font-family: 'JetBrains Mono', monospace;")
        layout.addWidget(self.table_label)

        self.setLayout(layout)

    def update_info(
        self,
        x: int,
        y: int,
        slice_idx: int,
        params: np.ndarray,
    ) -> None:
        """Update the displayed coordinate/parameter information."""
        self.coord_label.setText(f"Voxel: x={x}, y={y}, slice={slice_idx}")
        rows = []
        for name, value in zip(self.parameter_names, params):
            if np.isnan(value):
                continue
            rows.append(
                f"<tr><td style='padding-right:12px;'>{name.upper()}</td>"
                f"<td style='text-align:right;'>{value:.2f}</td></tr>"
            )
        if not rows:
            rows_html = "<tr><td colspan='2'>No fit available</td></tr>"
        else:
            rows_html = "".join(rows)
        html = (
            "<table style='width:100%; font-size:12px;'>"
            "<tbody>"
            f"{rows_html}"
            "</tbody>"
            "</table>"
        )
        self.table_label.setText(html)


class ROISummaryPanel(QWidget):
    """Panel showing ROI statistics: pixels, mean, std per unique label."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("ROI SUMMARY")
        title.setStyleSheet("font-weight: 600; color: #7cd5ff;")
        layout.addWidget(title)

        self.table_label = QLabel("<i>No mask loaded</i>")
        self.table_label.setWordWrap(True)
        self.table_label.setTextFormat(Qt.RichText)
        self.table_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.table_label.setStyleSheet("font-family: 'JetBrains Mono', monospace; font-size: 11px;")
        layout.addWidget(self.table_label)

        self.setLayout(layout)

    def update_summary(self, fit_maps: np.ndarray, param_index: int, mask_data: np.ndarray | None) -> None:
        """Calculate and display ROI statistics per label."""
        if fit_maps is None:
            self.table_label.setText("<i>No data</i>")
            return

        param_data = fit_maps[param_index]

        if mask_data is None:
            # Fallback: show total stats if no mask
            valid_mask = (param_data > 0) & np.isfinite(param_data)
            if not np.any(valid_mask):
                self.table_label.setText("<i>No valid pixels</i>")
                return
            valid_values = param_data[valid_mask]
            html = (
                "<table style='width:100%;'>"
                "<tr style='color:#888;'><td>ROI</td><td>Pixels</td><td>Mean</td><td>Std</td></tr>"
                "<tr><td>All</td><td>{}</td><td>{:.1f}</td><td>{:.1f}</td></tr>"
                "</table>"
            ).format(int(np.sum(valid_mask)), np.nanmean(valid_values), np.nanstd(valid_values))
            self.table_label.setText(html)
            return

        # Get unique ROI labels (excluding 0)
        unique_labels = np.unique(mask_data)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) == 0:
            self.table_label.setText("<i>No ROIs in mask</i>")
            return

        rows = ["<tr style='color:#888;'><td>ROI</td><td>Px</td><td>Mean</td><td>Std</td></tr>"]
        for label in sorted(unique_labels):
            roi_mask = (mask_data == label) & (param_data > 0) & np.isfinite(param_data)
            if not np.any(roi_mask):
                rows.append(f"<tr><td>{int(label)}</td><td>0</td><td>-</td><td>-</td></tr>")
                continue
            values = param_data[roi_mask]
            pixels = int(np.sum(roi_mask))
            mean_val = float(np.nanmean(values))
            std_val = float(np.nanstd(values))
            rows.append(f"<tr><td><b>{int(label)}</b></td><td>{pixels}</td><td>{mean_val:.1f}</td><td>{std_val:.1f}</td></tr>")

        html = "<table style='width:100%;'>" + "".join(rows) + "</table>"
        self.table_label.setText(html)


def example_1():
    import pydicom
    from pydicom.data import get_testdata_files

    filename = get_testdata_files("CT_small.dcm")[0]
    ds = pydicom.dcmread(filename)
    default = ds.pixel_array

    # Create a 3D DICOM array
    dicom = np.zeros((10, 128, 128, 42))
    for i in range(10):
        for ii in range(42):
            dicom[i, :, :, ii] = default * (i + 1)

    dicom = dicom / dicom.max() * 4016
    time_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Create fitted_map
    fit_maps = np.array(
        [
            (dicom[-1] - dicom[0]) / (time_points[-1] - time_points[0]),
            np.zeros_like(dicom[0]),
        ]
    )

    @njit
    def fit(x, a, b):
        return a * x + b

    # Create an instance of the ImageViewer class

    app = QApplication(sys.argv)

    viewer = ImageViewer()
    viewer.start(dicom, fit_maps, fit, time_points, 0, normalize=False)
    viewer.show()

    # Run the PyQt5 application
    sys.exit(app.exec_())


def example_2():
    from src.Fitting import T1rho_T2prep

    t1rho_folder = (
        Path(__file__).parent.parent.parent
        / "test"
        / "resources"
        / "20211206_1038"
        / "T1rho"
    )
    t1rho = T1rho_T2prep(dim=3)
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.start(
        dicom=t1rho_folder / "dicom.nii.gz",
        fit_maps=t1rho_folder / "params.nii.gz",
        fit_function=t1rho.fit_function,
        time_points=[0, 20, 80, 140],
        c_int=1,
    )
    viewer.show()
    sys.exit(app.exec_())


def example_t2star():
    from src.Fitting import T2_T2star

    t2_star_folder = (
        Path(__file__).parent.parent.parent
        / "test"
        / "resources"
        / "20211206_1038"
        / "7_T2-star_map_3D_cor_18818"
    )
    t2star = T2_T2star(dim=3)
    time_points = t2star.load_times(t2_star_folder / "acquisition_times.txt")
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.start(
        dicom=t2_star_folder / "dicom.nii.gz",
        fit_maps=t2_star_folder / "params.nii.gz",
        fit_function=t2star.fit_function,
        time_points=time_points,
        c_int=1,
    )
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    example_t2star()
