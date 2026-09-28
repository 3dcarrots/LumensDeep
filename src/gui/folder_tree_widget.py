import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeView, 
    QFileSystemModel, QHeaderView, QLineEdit, QPushButton, QLabel
)
from PySide6.QtCore import Qt, Signal, QDir, QModelIndex

class FolderTreeWidget(QWidget):
    """
    Folder Tree navigation widget showing full filesystem hierarchy,
    including hard drives, removable USB flash drives, and memory cards.
    """
    folder_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Top Bar with title & refresh
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(4)

        self.lbl_title = QLabel("📁 Folders")
        self.lbl_title.setStyleSheet("font-weight: bold; color: #ddd; font-size: 11px;")
        top_bar.addWidget(self.lbl_title)
        top_bar.addStretch()

        self.btn_refresh = QPushButton("🔄")
        self.btn_refresh.setToolTip("Refresh drives and folders")
        self.btn_refresh.setFixedSize(26, 24)
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #353535;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                color: #fff;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #454545;
            }
        """)
        self.btn_refresh.clicked.connect(self.refresh)
        top_bar.addWidget(self.btn_refresh)

        layout.addLayout(top_bar)

        # QFileSystemModel
        self.model = QFileSystemModel()
        self.model.setFilter(QDir.Dirs | QDir.NoDotAndDotDot | QDir.Drives)
        self.model.setRootPath("")

        # QTreeView
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setHeaderHidden(True)
        
        # Hide Size, Type, Date columns
        for col in range(1, 4):
            self.tree_view.hideColumn(col)

        self.tree_view.setAnimated(True)
        self.tree_view.setIndentation(16)
        self.tree_view.setUniformRowHeights(True)
        
        # Styling
        self.tree_view.setStyleSheet("""
            QTreeView {
                background-color: #202020;
                border: 1px solid #333333;
                border-radius: 4px;
                color: #e0e0e0;
                padding: 2px;
                font-size: 11px;
            }
            QTreeView::item {
                padding: 3px 2px;
                border-radius: 3px;
            }
            QTreeView::item:hover {
                background-color: #2e2e2e;
            }
            QTreeView::item:selected {
                background-color: #1f4f82;
                color: #ffffff;
            }
        """)

        self.tree_view.clicked.connect(self._on_tree_clicked)
        self.tree_view.activated.connect(self._on_tree_clicked)
        if self.tree_view.selectionModel():
            self.tree_view.selectionModel().currentChanged.connect(self._on_current_changed)
        layout.addWidget(self.tree_view)

    def _on_current_changed(self, current: QModelIndex, previous: QModelIndex):
        self._on_tree_clicked(current)

    def _on_tree_clicked(self, index: QModelIndex):
        if not index.isValid():
            return
        path = self.model.filePath(index)
        if path and os.path.exists(path) and os.path.isdir(path):
            self.folder_selected.emit(path)

    def select_folder(self, folder_path: str):
        """Expand tree to and select the given directory path."""
        if not folder_path or not os.path.exists(folder_path):
            return
            
        norm_path = os.path.abspath(folder_path)
        index = self.model.index(norm_path)
        if not index.isValid():
            # If model hasn't loaded this path yet, set root path or fetch
            self.model.setRootPath(norm_path)
            index = self.model.index(norm_path)

        if index.isValid():
            self.tree_view.setCurrentIndex(index)
            self.tree_view.scrollTo(index)
            self.tree_view.expand(index)

    def refresh(self):
        """Refresh folder tree model."""
        self.model.setRootPath("")
        current_idx = self.tree_view.currentIndex()
        if current_idx.isValid():
            self.tree_view.scrollTo(current_idx)
