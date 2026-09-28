from PySide6.QtWidgets import QDockWidget, QSizePolicy
from PySide6.QtCore import QSize
from ..filmstrip_widget import FilmstripWidget

class FilmstripDock(QDockWidget):
    """
    Bottom dock wrapper containing the Filmstrip thumbnail bar.
    """
    def __init__(self, main_window, parent=None):
        super().__init__("Filmstrip", parent)
        self.main_window = main_window
        self.filmstrip_widget = FilmstripWidget(parent=self)
        self.setWidget(self.filmstrip_widget)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def sizeHint(self):
        return QSize(400, 160)

    def minimumSizeHint(self):
        return QSize(100, 90)
