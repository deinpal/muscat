import os
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from .muscat_core import run_markov_prediction


class MUSCATDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(MUSCATDialog, self).__init__(parent)
        from .ui_muscat import Ui_MUSCATDialog
        self.ui = Ui_MUSCATDialog()
        self.ui.setupUi(self)

        # Connect buttons
        self.ui.btnBrowseInitial.clicked.connect(self.browse_initial)
        self.ui.btnBrowseFinal.clicked.connect(self.browse_final)
        self.ui.btnBrowseOut.clicked.connect(self.browse_output)
        self.ui.btnAddDriver.clicked.connect(self.add_driver)
        self.ui.btnRun.clicked.connect(self.run_model)

    def browse_initial(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Initial Raster", "", "Raster Files (*.tif *.tiff)")
        if path:
            self.ui.lineInitial.setText(path)

    def browse_final(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Final Raster", "", "Raster Files (*.tif *.tiff)")
        if path:
            self.ui.lineFinal.setText(path)

    def browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Output Raster", "", "GeoTIFF (*.tif)")
        if path:
            self.ui.lineOut.setText(path)

    def add_driver(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Spatial Variable Raster", "", "Raster Files (*.tif *.tiff)")
        if path:
            current = self.ui.textDrivers.toPlainText()
            updated = current + ("\n" if current else "") + path
            self.ui.textDrivers.setPlainText(updated)

    def run_model(self):
        """Handles both simulation and validation run modes"""
        initial = self.ui.lineInitial.text()
        final = self.ui.lineFinal.text()
        out = self.ui.lineOut.text()
        steps = self.ui.spinSteps.value()
        run_mode = self.ui.comboRunMode.currentText()

        if not initial or not final or not out:
            QMessageBox.warning(self, "Missing Inputs", "Please select Initial, Final, and Output raster paths.")
            return

        drivers_text = self.ui.textDrivers.toPlainText().strip()
        drivers = drivers_text.splitlines() if drivers_text else None

        validation = None
        if run_mode == "Validate":
            val_path, _ = QFileDialog.getOpenFileName(self, "Select Validation Raster", "", "Raster Files (*.tif *.tiff)")
            if not val_path:
                QMessageBox.warning(self, "Validation Missing", "You must select a validation raster.")
                return
            validation = val_path

        try:
            result_raster, trans_matrix, area_table, val_results = run_markov_prediction(
                initial=initial,
                final=final,
                out=out,
                steps=steps,
                drivers=drivers,
                validation=validation,
                export_dir=None,
                progress_callback=self.update_progress
            )

            # Automatically export transition matrix and area change table
            out_dir = os.path.dirname(out)
            pd.DataFrame(trans_matrix).to_csv(os.path.join(out_dir, "transition_matrix.csv"), index=True)
            pd.DataFrame(area_table).to_csv(os.path.join(out_dir, "area_change_table.csv"), index=True)

            msg = "Simulation finished.\n\nTables exported in output folder.\n"
            if validation:
                msg += f"Accuracy: {val_results['Overall Accuracy']:.3f}\n"
                msg += f"Kappa: {val_results['Kappa']:.3f}\n"
            QMessageBox.information(self, "Run Complete", msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")

    def update_progress(self, value):
        """Update progress bar in UI"""
        self.ui.progressBar.setValue(int(value))