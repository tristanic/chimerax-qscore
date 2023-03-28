from chimerax.ui.gui import MainToolWindow

from Qt.QtWidgets import (
    QFrame, QLabel,
    QPushButton, QMenu, QRadioButton, QScrollBar,
    QHBoxLayout, QVBoxLayout,
)
from Qt import QtCore
from Qt.QtCore import Qt

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
        self._query_atoms = None

        layout = self.main_layout = DefaultVLayout()
        self.setLayout(layout)
        bf = self.top_button_frame = QFrame()
        layout.addWidget(bf)
        bl = DefaultHLayout()
        bf.setLayout(bl)
        bl.addWidget(QLabel('Atomic model: '))
        asb = AtomicStructureMenuButton(session, self)
        bl.addWidget(asb)
        bl.addWidget(QLabel('Chain: '))
        cb = self.chain_button = ChainChooserButton()
        self.triggers.add_handler('selected model changed', cb._selected_model_changed_cb)
        cb.triggers.add_handler('selected chain changed', self.update_plot)
        bl.addWidget(cb)
        bl.addStretch()
        bl.addWidget(QLabel('Map: '))
        vb = VolumeMenuButton(session, self)
        bl.addWidget(vb)

        pw = self.plot_widget = QScorePlot()
        layout.addWidget(pw)

        ptl = DefaultHLayout()
        ptt = self._plot_text_data = QLabel()
        ptl.addWidget(ptt)
        layout.addLayout(ptl)
        pw.initialize_hover_text(ptt)

        rbl = DefaultHLayout()
        rb = self.recalc_button = QPushButton('Calculate')
        rbl.addWidget(rb)
        ms = self.mode_selector = AverageModeChooser()
        rbl.addWidget(ms)
        ms.triggers.add_handler('mode changed', self.update_plot)
        rbl.addStretch()
    
        rb.clicked.connect(self.recalc)
        layout.addLayout(rbl)

    def recalc(self, *_):
        m, v = self.selected_model, self.selected_volume
        if m is None or v is None:
            from chimerax.core.errors import UserError
            raise UserError('Must select a model and map first!')
        from chimerax.core.commands import run
        residue_map, (query_atoms, atom_scores) = run(self.session, f'qscore #{m.id_string} to #{v.id_string} useGui false', log=False)
        self._residue_map = residue_map
        self._atom_scores = atom_scores
        self._query_atoms = query_atoms
        self.update_plot()
        return residue_map, (query_atoms, atom_scores)

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
        if model is not None and model != self._selected_model:
            from .clipper_compat import model_managed_by_clipper
            if not model_managed_by_clipper(model):
                session = model.session
                from chimerax.core.commands import run
                run(session, f'style #{model.id_string} stick; color #{model.id_string} byhet')
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
        if v is not None:
            from .clipper_compat import map_associated_with_model
            if not map_associated_with_model(self.selected_model, v):
                if any([s.display_style=='solid' for s in v.surfaces]):
                    from chimerax.core.commands import run
                    run (self.session, f'transparency #{v.id_string} 60')
        self.triggers.activate_trigger('selected volume changed', v)

    def clear_scores(self):
            self._residue_map = None
            self._atom_scores = None
            self.update_plot()


    def update_plot(self, *_):
        pw = self.plot_widget
        if self.selected_model is None or self.selected_volume is None or self._residue_map is None:
            pw.update_data(None, None, None)
            return
        import numpy
        cid = self.chain_button.selected_chain_id
        average_mode = self.mode_selector.mode
        residues = [r for r in self._residue_map.keys() if r.chain_id==cid]
        atoms = self._query_atoms
        chain_mask = atoms.residues.chain_ids==cid
        ascores = self._atom_scores
        from chimerax.atomic import Residue, Residues
        residues = Residues(residues)
        if average_mode == 'Whole residues':
            atom_mask = None
            scores = [self._residue_map[r][0] for r in residues]
        elif average_mode == 'Worst atoms':
            atom_mask = None
            scores = [self._residue_map[r][1] for r in residues]
        elif average_mode == 'Backbone':
            atom_mask = atoms.is_backbones()
        elif average_mode == 'Sidechains':
            atom_mask = atoms.is_side_onlys
        elif average_mode == 'Ligands':
            atom_mask = atoms.residues.polymer_types == Residue.PT_NONE
        else:
            raise RuntimeError(f'Unrecognised averaging mode: "{average_mode}"')
        if atom_mask is not None:
            atoms = atoms[numpy.logical_and(atom_mask, chain_mask)]
            ascores = ascores[atom_mask]
            residues = atoms.unique_residues
            scores = []
            for r in residues:
                indices = atoms.indices(r.atoms)
                indices = indices[indices!=-1]
                scores.append(ascores[indices].mean())            
        pw.update_data(residues, scores, self.selected_volume)

    def cleanup(self):
        while len(self._handlers):
            h = self._handlers.pop()
            h.remove()

class AverageModeChooser(QFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger('mode changed')
        layout = DefaultHLayout()
        self.setLayout(layout)
        tooltips = {
            'Whole residues': '<span>Mean Q-score for all non-hydrogen atoms in a each residue.</span>',
            'Worst atoms': '<span>Q-score for the worst non-hydrogen atom in each residue.</span>',
            'Backbone': '<span>Mean Q-score for non-hydrogen backbone atoms in each residue. Excludes ligands.</span>',
            'Sidechains': '<span>Mean Q-score for non-hydrogen sidechain atoms in each residue. Excludes ligands.<span>',
            'Ligands': '<span>Mean Q-score for non-hydrogen atoms in each ligand.</span>'
        }
        self.modes = {
            'Whole residues': QRadioButton('Whole residues'),
            'Worst atoms': QRadioButton('Worst atoms'),
            'Backbone': QRadioButton('Backbone'),
            'Sidechains': QRadioButton('Sidechains'),
            'Ligands': QRadioButton('Ligands')
        }
        for key, button in self.modes.items():
            button.setToolTip(tooltips[key])
        self.modes['Whole residues'].setChecked(True)
        for mb in self.modes.values():
            layout.addWidget(mb)
            def _cb(mode):
                for mname, mde in self.modes.items():
                    if mde == mode:
                        break
                self.triggers.activate_trigger('mode changed', mname)
            mb.toggled.connect(lambda:_cb(mb))
    
    @property
    def mode(self):
        for mode, button in self.modes.items():
            if button.isChecked():
                return mode
    



class QScorePlot(QFrame):
    MIN_ZOOM = 25
    DEFAULT_ZOOM = 250
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
        self.setMinimumHeight(150)
        self._slider_blocked = False
        ml = self.main_layout = DefaultVLayout()
        self.setLayout(ml)

        fig = self.plot = Figure()
        axes = self.axes = fig.add_subplot(111)
        axes.autoscale(enable=False)
        axes.set_ylim(0,1)
        axes.set_xlim(0, self.DEFAULT_ZOOM)
        fig.subplots_adjust(bottom=0.25)

        self.volume = None

        fig.tight_layout(rect=(0,0.1,1,1))

        self.residues = []
        resnum = self.residue_numbers = numpy.zeros(2, dtype=numpy.int32)-1
        qscore = self.qscores = numpy.array([0,1], dtype=numpy.float32)
        def _picker(scatter, mouse_event, x_radius=1):
            from matplotlib.backend_bases import MouseButton
            if mouse_event.button != MouseButton.LEFT:
                return False, dict()
            import numpy
            if mouse_event.inaxes != self.axes:
                return False, dict()
            if mouse_event.xdata is None:
                return False, dict()
            x = mouse_event.xdata
            resnums = scatter.get_offsets().T[0]
            d = numpy.sqrt((x - resnums)**2)
            closest = d.min()
            if closest > x_radius:
                return False, dict()
            ind = numpy.argmin(d)
            return True, dict(ind=ind)
            
            
        s = axes.scatter(resnum,qscore, s=4, c=qscore, cmap='inferno_r', picker=_picker)
        self._scatter = s

        canvas = self.canvas = FigureCanvas(fig)
        
        canvas.mpl_connect('scroll_event', self.zoom)
        canvas.mpl_connect('pick_event', self.on_pick)

        ml.addWidget(canvas)

        hpos = self._hpos_slider = QScrollBar(Qt.Orientation.Horizontal)
        hpos.setRange(0,0)
        ml.addWidget(hpos)

        hpos.valueChanged.connect(self._slider_update)

        canvas.draw()

    def initialize_hover_text(self, target):
        self._hover_text_target = target

        def _hover(event, x_radius=1):
            if not len(self.residues):
                target.setText('')
                return
            if event.inaxes != self.axes:
                target.setText('')
                return
            import numpy
            x = event.xdata
            resnums = self._scatter.get_offsets().T[0]
            d = numpy.sqrt((x-resnums)**2)
            closest = d.min()
            if closest > x_radius:
                target.setText('')
                return
            ind = numpy.argmin(d)
            r = self.residues[ind]
            if r.deleted:
                target.setText('{residue deleted}')
                return
            target.setText(f'{r.name} /{r.chain_id}:{r.number}\tQ: {self.qscores[ind]:.3f}')
            return
        self.canvas.mpl_connect('motion_notify_event', _hover)


    def _slider_update(self, val):
        if not len(self.residues) or self._slider_blocked:
            return
        axes = self.axes
        xmin, xmax = axes.get_xlim()
        xrange = xmax-xmin
        axes.set_xlim([val,val+xrange])
        self.canvas.draw_idle()

    def zoom(self, event=None):
        if not len(self.residues):
            return
        axes = self.axes
        hpos = self._hpos_slider
        xmin, xmax = axes.get_xlim()            
        xrange = xmax-xmin
        resnum = self.residue_numbers
        if xmax > resnum.max():
            xmax = resnum.max()
            xmin = xmax-xrange
        if xmin < resnum.min():
            xmin = resnum.min()
            xmax = xmin+xrange

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
        if xrange >= resnum.max()-resnum.min():
            hpos.setRange(resnum.min(), resnum.min())
        else:
            with slot_disconnected(hpos.valueChanged, self._slider_update):
                hpos.setRange(resnum.min(), resnum.max()-xrange)
                hpos.setValue(xmin)
        self.canvas.draw_idle()

    def zoom_extents(self):
        resnum = self.residue_numbers
        self.axes.set_xlim([resnum.min(), resnum.max()])
        self.zoom()

    def on_pick(self, event):
        if not len(self.residues):
            return
        ind = event.ind
        residue = self.residues[ind]
        if residue.deleted:
            return
        session = residue.session
        session.selection.clear()
        residue.atoms.selected = True
        residue.atoms.intra_bonds.selected = True
        atomspec = f'#!{residue.structure.id_string}/{residue.chain_id}:{residue.number}'
        from chimerax.core.commands import run
        from .clipper_compat import model_managed_by_clipper
        m = residue.structure

        if model_managed_by_clipper(m):
            # Just view the model
            run(session, f'view {atomspec}')
        else:
            # TODO: decide what to do here
            from chimerax.atomic import Residues, concise_residue_spec
            neighbors = set([residue])
            # Quick and (very) dirty way to expand the selection. Should probably do something more efficient.
            for _ in range(3):
                new_neighbors = []
                for n in neighbors:
                    for nn in n.neighbors:
                        if nn not in neighbors:
                            new_neighbors.append(nn)
                neighbors.update(new_neighbors)
            residues = Residues(neighbors)
            argspec = concise_residue_spec(session, residues)
            run(session, f'surf zone #{self.volume.id_string} near {argspec} dist 3', log=False)
            run(session, f'~cartoon #{m.id_string}; hide #{m.id_string}; show {argspec}; cartoon {argspec}&~{atomspec}', log=False)
            run(session, f'view {atomspec}', log=False)            

    def update_data(self, residues, scores, volume):
        self.volume = volume
        if residues is None or not len(residues):
            self.residues = []
            self._scatter.set_offsets([[0,0]])
        else:
            # Convert to a list so we can gracefully handle residue deletions in callbacks
            import numpy
            update_zoom = False
            if len(residues) != len(self.residues):
                update_zoom = True
            self.residues = list(residues)
            rmin = min(r.number for r in residues)
            rmax = max(r.number for r in residues)
            resnum = self.residue_numbers = numpy.array([r.number for r in residues])
            self.qscores = scores
            clipped_scores = numpy.array(scores, copy=True)
            clipped_scores[clipped_scores<0] = 0
            self._scatter.set_offsets(numpy.array([resnum, clipped_scores]).T)
            self._scatter.set_array(scores)
            # If the number of plotted residues changes, zoom to the full extent of the data.
            if update_zoom:
                self.zoom_extents()
            else:
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
        if m is None:
            self.selected_chain_id = None
            return
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
    DEFAULT_TOOLTIP=''
    def __init__(self, session, owner,  *args, model_type=None, trigger_name=None, **kwargs):
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
        self.setToolTip(self.DEFAULT_TOOLTIP)
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
            self.setToolTip(self.DEFAULT_TOOLTIP)
        else:
            import textwrap
            self.setText(f'#{model.id_string}: {textwrap.shorten(model.name, 12)}')
            self.setToolTip(f'<span>#{model.id_string}: {model.name}</span>')

class AtomicStructureMenuButton(ModelMenuButtonBase):
    DEFAULT_TOOLTIP='Atomic model to use for Q-score calculation.'
    def __init__(self, session, owner, *args, **kwargs):
        from chimerax.atomic import AtomicStructure
        super().__init__(session, owner, *args, 
                        model_type=AtomicStructure,
                        trigger_name='selected model changed', 
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
    DEFAULT_TOOLTIP='Volume to use for Q-score calculation.'
    def __init__(self, session, owner, *args, **kwargs):
        from chimerax.map import Volume
        super().__init__(session, owner, *args, 
                         model_type=Volume,
                         trigger_name='selected volume changed',
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
        if not any([len(mlist) for mlist in (assoc, free, other)]):
            return
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
            


        

from contextlib import contextmanager
@contextmanager
def slot_disconnected(signal, slot):
    '''
    Temporarily disconnect a slot from a signal using

    .. code-block:: python
    
        with slot_disconnected(signal, slot):
            do_something()
    
    The signal is guaranteed to be reconnected even if do_something() throws an error.
    '''
    try:
        # disconnect() throws a TypeError if the method is not connected
        signal.disconnect(slot)
        yield
    except TypeError:
        pass
    finally:
        signal.connect(slot)