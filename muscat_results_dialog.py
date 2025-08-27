from qgis.PyQt import QtWidgets
import pandas as pd

class MUSCATResultsDialog(QtWidgets.QDialog):
    def __init__(self, trans_df: pd.DataFrame, area_df: pd.DataFrame, validation_results=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MUSCAT – Results Preview")
        self.resize(750, 550)

        tabs = QtWidgets.QTabWidget(self)

        # --- Transition Matrix Tab ---
        self.tableTrans = QtWidgets.QTableWidget()
        self._populate_table(self.tableTrans, trans_df)
        tabs.addTab(self.tableTrans, "Transition Matrix")

        # --- Area Change Tab ---
        self.tableArea = QtWidgets.QTableWidget()
        self._populate_table(self.tableArea, area_df)
        tabs.addTab(self.tableArea, "Area Change Table")

        # --- Validation Tab (Optional) ---
        if validation_results is not None:
            validation_widget = QtWidgets.QWidget()
            vlayout = QtWidgets.QVBoxLayout(validation_widget)

            # Metrics
            lbl_metrics = QtWidgets.QLabel(
                f"<b>Overall Accuracy:</b> {validation_results['Overall Accuracy']:.3f}<br>"
                f"<b>Kappa:</b> {validation_results['Kappa']:.3f}"
            )
            vlayout.addWidget(lbl_metrics)

            # Confusion Matrix Table
            cm_df = pd.DataFrame(
                validation_results["Confusion Matrix"],
                index=trans_df.index,
                columns=trans_df.columns
            )
            cm_table = QtWidgets.QTableWidget()
            self._populate_table(cm_table, cm_df)
            vlayout.addWidget(cm_table)

            tabs.addTab(validation_widget, "Validation")

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)

    def _populate_table(self, table: QtWidgets.QTableWidget, df: pd.DataFrame):
        """Fill QTableWidget with pandas DataFrame."""
        table.setRowCount(len(df.index))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        table.setVerticalHeaderLabels([str(i) for i in df.index])

        for i, idx in enumerate(df.index):
            for j, col in enumerate(df.columns):
                val = df.iloc[i, j]
                item = QtWidgets.QTableWidgetItem(str(val))
                table.setItem(i, j, item)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()