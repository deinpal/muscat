def classFactory(iface):
    from .muscat_plugin import MUSCATPlugin
    return MUSCATPlugin(iface)