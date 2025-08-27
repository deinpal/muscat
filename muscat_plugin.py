from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon
from .muscat_dialog import MUSCATDialog

class MUSCATPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dlg = None

    def initGui(self):
        self.action = QAction(QIcon(), "MUSCAT", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&MUSCAT", self.action)

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&MUSCAT", self.action)

    def run(self):
        if not self.dlg:
            self.dlg = MUSCATDialog()
        self.dlg.show()
        self.dlg.exec_()
