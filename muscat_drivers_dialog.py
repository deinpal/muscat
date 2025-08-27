import os
from qgis.PyQt import QtWidgets, uic
from qgis.PyQt.QtWidgets import QFileDialog

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), "ui_muscat_drivers.ui")
)

class MUSCATDriversDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super(MUSCATDriversDialog, self).__init__(parent)
        self.setupUi(self)

        # Connect button
        self.btnAddDriver.clicked.connect(self.add_driver)

    def add_driver(self):
        """Open file dialog to add a raster driver"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Driver Raster",
            "",
            "Raster files (*.tif *.img)"
        )
        if path:
            text = self.textDrivers.toPlainText().strip()
            if text:
                text += f"\n{path}|1.0"
            else:
                text = f"{path}|1.0"
            self.textDrivers.setPlainText(text)

    def get_drivers(self):
        """
        Returns a list of (path, weight) tuples from the text box.
        Default weight is 1.0 if not specified.
        """
        drivers = []
        lines = self.textDrivers.toPlainText().strip().splitlines()
        for line in lines:
            if "|" in line:
                parts = line.split("|")
                try:
                    weight = float(parts[1])
                except ValueError:
                    weight = 1.0
                drivers.append((parts[0], weight))
            else:
                drivers.append((line, 1.0))
        return drivers