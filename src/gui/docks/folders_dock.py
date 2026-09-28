from PySide6.QtWidgets import QDockWidget
from ..folder_tree_widget import FolderTreeWidget

class FoldersDock(QDockWidget):
    """
    Left dock wrapper containing the Folder filesystem navigation tree.
    """
    def __init__(self, main_window, parent=None):
        super().__init__("Folders", parent)
        self.main_window = main_window
        self.folder_tree = FolderTreeWidget(parent=self)
        self.setWidget(self.folder_tree)
