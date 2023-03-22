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
        session.qscorewidget = self # DEBUG
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

        self._residue_map = None
        self._atom_scores = None

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

        rbl = DefaultHLayout()
        rb = self.recalc_button = QPushButton('Recalculate')
        rbl.addWidget(rb)
        rbl.addStretch()
        rbl.addWidget(QLabel('Chain: '))
        cb = self.chain_button = ChainChooserButton()
        self.triggers.add_handler('selected model changed', cb._selected_model_changed_cb)
        cb.triggers.add_handler('selected chain changed', self.update_plot)
        rbl.addWidget(cb)

        def _recalc(*_):
            m, v = self.selected_model, self.selected_volume
            if m is None or v is None:
                from chimerax.core.errors import UserError
                raise UserError('Must select a model and map first!')
            from chimerax.core.commands import run
            residue_map, atom_scores = run(session, f'qscore #{m.id_string} to #{v.id_string}')
            self._residue_map = residue_map
            self._atom_scores = atom_scores
            self.update_plot()
        rb.clicked.connect(_recalc)
        layout.addLayout(rbl)


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
        if self._selected_model != model:
            self.clear_scores()
        self._selected_model = model
        self.triggers.activate_trigger('selected model changed', model)
    
    @property
    def selected_volume(self):
        return self._selected_volume
    
    @selected_volume.setter
    def selected_volume(self, v):
        if self._selected_volume != v:
            self.clear_scores()
        self._selected_volume = v
        self.triggers.activate_trigger('selected volume changed', v)

    def clear_scores(self):
            self._residue_map = None
            self._atom_scores = None
            self.update_plot()


    def update_plot(self, *_):
        pw = self.plot_widget
        if self.selected_model is None or self.selected_volume is None or self._residue_map is None:
            pw.update_data(None, None)
            return
        cid = self.chain_button.selected_chain_id
        residues = [r for r in self._residue_map.keys() if r.chain_id==cid]
        scores = [self._residue_map[r][0] for r in residues]
        pw.update_data(residues, scores)



    def cleanup(self):
        while len(self._handlers):
            h = self._handlers.pop()
            h.remove()

class QScorePlot(QFrame):
    MIN_ZOOM = 25
    DEFAULT_ZOOM = 100
    ZOOM_SCALE = 1.1
    MAX_ZOOM_STEPS_PER_FRAME=5
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvas,
            )
        from matplotlib.widgets import Slider
        from matplotlib.colors import Normalize
        import numpy

        self._slider_blocked = False
        ml = self.main_layout = DefaultVLayout()
        self.setLayout(ml)

        fig = self.plot = Figure()
        axes = self.axes = fig.add_subplot(111)
        axes.autoscale(enable=False)
        axes.set_ylim(-1,1)
        axes.set_xlim(0, self.DEFAULT_ZOOM)
        fig.subplots_adjust(bottom=0.25)


        sax = fig.add_axes([0.2, 0.1, 0.65, 0.03])
        hpos = self._hpos_slider = Slider(sax, '', 0, 1, valinit=0)
        hpos.valtext.set_visible(False)

        self.residues = []
        resnum = self.residue_numbers = numpy.zeros(2, dtype=numpy.int32)-1
        qscore = self.qscores = numpy.array([0,1], dtype=numpy.float32)
        s = axes.scatter(resnum,qscore, s=4, c=qscore, cmap='inferno_r')
        self._scatter = s

        canvas = self.canvas = FigureCanvas(fig)
        
        canvas.mpl_connect('scroll_event', self.zoom)

        hpos.on_changed(self._slider_update)

        ml.addWidget(canvas)
        canvas.draw()

    def _slider_update(self, val):
        if not len(self.residues) or self._slider_blocked:
            return
        axes = self.axes
        hpos = self._hpos_slider
        pos = val
        xmin, xmax = axes.get_xlim()
        xrange = xmax-xmin
        resnum = self.residue_numbers
        new_xmin = pos * (resnum.max()-xrange-resnum.min()) + resnum.min()

        axes.axis([new_xmin,new_xmin+xrange,-1,1])
        self.canvas.draw_idle()
    
    def zoom(self, event=None):
        if not len(self.residues):
            return
        axes = self.axes
        hpos = self._hpos_slider
        xmin, xmax = axes.get_xlim()
        xrange = xmax-xmin
        resnum = self.residue_numbers
        if event is not None:
            if event.inaxes != self.axes:
                return
            xpoint = event.xdata
            xfrac = (xpoint-xmin)/(xmax-xmin)
            if event.button == 'up':
                if xrange <= self.MIN_ZOOM:
                    return
                xrange = int(xrange/(self.ZOOM_SCALE*min(event.step, self.MAX_ZOOM_STEPS_PER_FRAME)))
            else:
                if xrange >= resnum.max()-resnum.min():
                    return
                xrange = int(xrange*self.ZOOM_SCALE*-min(event.step, self.MAX_ZOOM_STEPS_PER_FRAME))
            new_xmin = int(max(min(resnum), min(resnum.max()-xrange, int(xpoint-xrange*xfrac))))
        else:
            new_xmin = max(resnum.min(), xmin)
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



    def update_data(self, residues, scores):
        if residues is None:
            self.residues = []
            self._scatter.set_offsets([[0,0]])
        else:
            # Convert to a list so we can gracefully handle residue deletions in callbacks
            import numpy
            self.residues = list(residues)
            rmin = min(r.number for r in residues)
            rmax = max(r.number for r in residues)
            resnum = self.residue_numbers = numpy.array([r.number for r in residues])
            self.qscores = scores
            self._scatter.set_offsets(numpy.array([resnum, scores]).T)
            self._scatter.set_array(scores)
            # Update the plot limits if necessary
            self.zoom(None)
        self.canvas.draw_idle()

class ChainChooserButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_model = None
        cm = self.chain_menu = QMenu()
        self.setMenu(cm)
        cm.aboutToShow.connect(self._populate_available_chains_menu)
        from chimerax.core.triggerset import TriggerSet
        t = self.triggers = TriggerSet()
        t.add_trigger('selected chain changed')
        self.selected_chain_id = None

    def _selected_model_changed_cb(self, trigger_name, m):
        self.selected_model = m
        cids = m.residues.unique_chain_ids
        if self.selected_chain_id is None or self.selected_chain_id not in cids:
            self.selected_chain_id = cids[0]

    @property
    def selected_chain_id(self):
        return self._selected_chain_id
    
    @selected_chain_id.setter
    def selected_chain_id(self, cid):
        self._selected_chain_id = cid
        if cid is None:
            self.setText('(None)')
        else:
            self.setText(cid)
        self.triggers.activate_trigger('selected chain changed', cid)
        

    def _populate_available_chains_menu(self):
        cm = self.chain_menu
        cm.clear()
        m = self.selected_model
        if m is None:
            self.selected_chain_id = None
            return
        cids = m.residues.unique_chain_ids
        for cid in cids:
            a = cm.addAction(cid)
            def _cb(*_, c = cid):
                self.selected_chain_id = c
            a.triggered.connect(_cb)
    

            





class ModelMenuButtonBase(QPushButton):
    def __init__(self, session, owner,  *args, model_type=None, trigger_name=None, tooltip='', **kwargs):
        super().__init__(*args, **kwargs)
        self.session = session
        self.owner = owner
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
            
            



        


