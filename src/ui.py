from chimerax.ui.gui import MainToolWindow

from Qt.QtWidgets import (
    QWidget, QFrame, QScrollArea, QLabel,
    QPushButton, QMenu,
    QHBoxLayout, QVBoxLayout,
    QSpinBox, QDoubleSpinBox
)
from Qt import QtCore


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
        parent.setStyleSheet('')
        main_layout = DefaultVLayout()
        parent.setLayout(main_layout)
        self.main_widget = QScoreWidget(self.session, self)
        main_layout.addWidget(self.main_widget)
        self.manage(placement='side')
    
    def cleanup(self):
        while len(self._registered_widgets):
            w = self._registered_widgets.pop()
            w.cleanup()
        
    def register_widget(self, w):
        self._registered_widgets.append(w)


class QScoreWidget(QFrame):
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
        bf = self.top_button_frame = QFrame()
        layout.addWidget(bf)
        bl = DefaultHLayout()
        bf.setLayout(bl)
        bl.addWidget(QLabel('Atomic model: '))
        asb = AtomicStructureMenuButton(session, self)
        bl.addWidget(asb)
        bl.addStretch()
        bl.addWidget(QLabel('Map: '))
        vb = VolumeMenuButton(session, self)
        bl.addWidget(vb)

        pw = self.plot_widget = QScorePlot()
        layout.addWidget(pw)


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

class QScorePlot(QFrame):
    MIN_ZOOM = 25
    ZOOM_SCALE = 1.1
    MAX_ZOOM_STEPS_PER_FRAME=5
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvas,
            )
        from matplotlib.widgets import Slider

        self._slider_blocked = False
        ml = self.main_layout = DefaultVLayout()
        self.setLayout(ml)

        fig = self.plot = Figure()
        axes = self.axes = fig.add_subplot(111)
        axes.autoscale(enable=False)
        axes.set_ylim(-1,1)
        axes.set_xlim(0, self.MIN_ZOOM)
        fig.subplots_adjust(bottom=0.25)

        sax = fig.add_axes([0.2, 0.1, 0.65, 0.03])
        hpos = Slider(sax, '', 0, 1, valinit=0)

        # TEST DATA
        import numpy as np
        resnum = np.arange(0.0, 100.0, 0.1)
        qscore = np.sin(2*np.pi*resnum)
        l, = axes.plot(resnum,qscore)
        # /TEST DATA

        canvas = self.canvas = FigureCanvas(fig)

        def slider_update(val):
            if self._slider_blocked:
                return
            pos = hpos.val
            xmin, xmax = axes.get_xlim()
            xrange = xmax-xmin
            new_xmin = pos * (resnum.max()-xrange-resnum.min())

            axes.axis([new_xmin,new_xmin+xrange,-1,1])
            canvas.draw_idle()
        
        def zoom(event):
            xmin, xmax = axes.get_xlim()
            xpoint = event.xdata
            xrange = xmax-xmin
            if event.button == 'up':
                if xrange <= self.MIN_ZOOM:
                    return
                xrange = int(xrange/(self.ZOOM_SCALE*min(event.step, self.MAX_ZOOM_STEPS_PER_FRAME)))
            else:
                if xrange >= resnum.max()-resnum.min():
                    return
                xrange = int(xrange*self.ZOOM_SCALE*-min(event.step, self.MAX_ZOOM_STEPS_PER_FRAME))
            new_xmin = min(resnum.max()-xrange, max(min(resnum), int(xpoint-xrange/2)))
            new_xmax = min(max(resnum), new_xmin+xrange)
            axes.set_xlim([new_xmin, new_xmax])
            self._slider_blocked = True
            if xrange >= resnum.max()-resnum.min():
                hpos.set_active(False)
                hpos.set_val(0)
            else:
                hpos.set_active(True)
                hpos.set_val((new_xmin-resnum.min())/((resnum.max()-xrange)-resnum.min()))
            self._slider_blocked = False
        
        canvas.mpl_connect('scroll_event', zoom)

                


        hpos.on_changed(slider_update)

        ml.addWidget(canvas)
        canvas.draw()


class ModelMenuButtonBase(QPushButton):
    def __init__(self, session, owner,  *args, model_type=None, trigger_name=None, tooltip='', **kwargs):
        super().__init__(*args, **kwargs)
        self.session = session
        self.owner = owner
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
            owner.triggers.add_handler(trigger_name, self._model_changed_cb)
    
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
    def __init__(self, session, owner, *args, **kwargs):
        from chimerax.atomic import AtomicStructure
        super().__init__(session, owner, *args, 
                        model_type=AtomicStructure,
                        trigger_name='selected model changed', 
                        tooltip='Atomic model to use for Q-score calculation.',
                        **kwargs)
        
    def _menu_entry_clicked(self, model=None):
        self.owner.selected_model = model

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
    def __init__(self, session, owner, *args, **kwargs):
        from chimerax.map import Volume
        super().__init__(session, owner, *args, 
                         model_type=Volume,
                         trigger_name='selected volume changed',
                         tooltip='Volume to use for Q-score calculation.',
                         **kwargs)
    
    def _find_available_models(self):
        volumes = super()._find_available_models()
        from .clipper_compat import model_managed_by_clipper, map_associated_with_model
        m = self.owner.selected_model
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
        self.owner.selected_volume=model

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
        if model_managed_by_clipper(self.owner.selected_model):
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
            
                
            

        


