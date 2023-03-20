from chimerax.core.tools import ToolInstance

class QScore_ToolUI(ToolInstance):
    SESSION_ENDURING = True

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)

        from .ui import QScoreWindow
        tw = self.tool_window = QScoreWindow(self)