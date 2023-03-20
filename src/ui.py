from chimerax.ui.gui import MainToolWindow

from Qt.QtWidgets import (
    QWidget, QFrame, QScrollArea, QLabel,
    QPushButton, QMenu,
    QHBoxLayout, QVBoxLayout,
    QSpinBox, QDoubleSpinBox
)
from Qt import QtCore

from matplotlib import pyplot as plt

class DefaultVLayout(QVBoxLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContentsMargins(0,2,0,0)
        self.setSpacing(3)

class DefaultHLayout(QHBoxLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContentsMargins(0,0,0,0)
        self.setSpacing(3)

class QScoreWindow(MainToolWindow):
    def __init__(self, tool_instance, **kw):
        super().__init__(tool_instance, **kw)
        self._registered_widgets = []
        parent = self.ui_area
        main_layout = DefaultVLayout()
        parent.setLayout(main_layout)
        self.main_widget = QScoreWidget(self.session, self)
        main_layout.addWidget(self.main_widget)
    
    def cleanup(self):
        while len(self._registered_widgets):
            w = self._registered_widgets.pop()
            w.cleanup()
        
    def register_widget(self, w):
        self._registered_widgets.append(w)



class QScoreWidget(QWidget):
    def __init__(self, session, main_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = session
        main_window.register_widget(self)
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger('selected model changed')
        self.triggers.add_trigger('selected volume changed')

        hl = self._handlers = []
        hl.append(session.triggers.add_handler('remove models', self._models_removed_cb))
        

        self._selected_model = None
        self._selected_volume = None

        layout = self.main_layout = DefaultVLayout()
        self.setLayout(layout)
        asb = AtomicStructureMenuButton(session, self)
        layout.addWidget(asb)
        vb = VolumeMenuButton(session, self)
        layout.addWidget(vb)

    def _models_removed_cb(self, trigger_name, removed):
        if self.selected_model in removed:
            self.selected_model = None
        if self.selected_volume in removed:
            self.selected_volume = None
        

    @property
    def selected_model(self):
        return self._selected_model

    @selected_model.setter
    def selected_model(self, model):
        self._selected_model = model
        self.triggers.activate_trigger('selected model changed', model)
    
    @property
    def selected_volume(self):
        return self._selected_volume
    
    @selected_volume.setter
    def selected_volume(self, v):
        self._selected_volume = v
        self.triggers.activate_trigger('selected volume changed', v)

    def cleanup(self):
        while len(self._handlers):
            h = self._handlers.pop()
            h.remove()


class ModelMenuButtonBase(QPushButton):
    def __init__(self, session, parent,  *args, model_type=None, trigger_name=None, tooltip='', **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.session = session
        self._selected_model = None
        if model_type is None:
            raise RuntimeError('Model type must be specified!')
        self.model_type = model_type
        mmm = self.master_model_menu = QMenu()
        mmm.aboutToShow.connect(self._populate_available_models_menu)
        self.setMenu(mmm)
        self.setMinimumSize(QtCore.QSize(150,0))
        self.setToolTip(tooltip)
        self._set_button_text(None)
        if trigger_name is not None:
            parent.triggers.add_handler(trigger_name, self._model_changed_cb)
    
    def _find_available_models(self):
        models = self.session.models.list(type=self.model_type)
        return sorted(models, key=lambda m:m.id)
    
    def _menu_entry_clicked(self, model=None):
        pass

    def _populate_available_models_menu(self):
        pass

    def _model_changed_cb(self, trigger_name, model):
        self._set_button_text(model)
    
    def _set_button_text(self, model):
        if model is None:
            self.setText('None')
        else:
            self.setText(f'#{model.id_string}')

class AtomicStructureMenuButton(ModelMenuButtonBase):
    def __init__(self, session, parent, *args, **kwargs):
        from chimerax.atomic import AtomicStructure
        super().__init__(session, parent, *args, 
                        model_type=AtomicStructure,
                        trigger_name='selected model changed', 
                        tooltip='Atomic model to use for Q-score calculation.',
                        **kwargs)
        
    def _menu_entry_clicked(self, model=None):
        self.parent().selected_model = model

    def _populate_available_models_menu(self):
        mmm = self.master_model_menu
        mmm.clear()
        models = self._find_available_models()
        for m in models:
            a = mmm.addAction(f'{m.id_string}: {m.name}')
            def _cb(_, model=m):
                self._menu_entry_clicked(model)
            a.triggered.connect(_cb)


class VolumeMenuButton(ModelMenuButtonBase):
    def __init__(self, session, parent, *args, **kwargs):
        from chimerax.map import Volume
        super().__init__(session, parent, *args, 
                         model_type=Volume,
                         trigger_name='selected volume changed',
                         tooltip='Volume to use for Q-score calculation.',
                         **kwargs)
    
    def _find_available_models(self):
        volumes = super()._find_available_models()
        from .clipper_compat import model_managed_by_clipper, map_associated_with_model
        m = self.parent().selected_model
        if m is not None and model_managed_by_clipper(m):
            from chimerax.clipper.maps import XmapHandler_Live, XmapHandler_Static, NXmapHandler
            assoc = []
            free = []
            other = []
            for v in volumes:
                if map_associated_with_model(m, v):
                    assoc.append(v)
                elif isinstance(v, (XmapHandler_Live, XmapHandler_Static, NXmapHandler)):
                    other.append(v)
                else:
                    free.append(v)
            return (assoc, free, other)
        return ([], volumes, [])
    
    def _menu_entry_clicked(self, model=None):
        self.parent().selected_volume=model

    def _populate_available_models_menu(self):
        mmm = self.master_model_menu
        mmm.clear()
        assoc, free, other = self._find_available_models()
        from .clipper_compat import model_managed_by_clipper
        def add_entry(v):
            a = mmm.addAction(f'{v.id_string}: {v.name}')
            def _cb(_, volume=v):
                self._menu_entry_clicked(volume)
            a.triggered.connect(_cb)
        if model_managed_by_clipper(self.parent().selected_model):
            if len(assoc):
                a = mmm.addAction('--- Associated maps ---')
                a.setToolTip('<span>Maps that have been assigned to the selected model by Clipper</span>')
                for v in assoc:
                    add_entry(v)
            if len(free):
                a = mmm.addAction('--- Free maps ---')
                a.setToolTip('<span>Maps that are not currently associated with any model.</span>')
                for v in free:
                    add_entry(v)
            if len(other):
                a = mmm.addAction('--- Other maps (CAUTION) ---')
                a.setToolTip('<span>Maps that are associated with a different atomic model by Clipper. '
                            'These should generally be avoided.</span>')
                for v in other:
                    add_entry(v)
        else:
            for v in free:
                add_entry(v)
            
                
            

        


