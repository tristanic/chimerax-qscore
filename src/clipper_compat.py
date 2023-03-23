def clipper_installed():
    try:
        from chimerax import clipper
        return True
    except ImportError:
        return False

def model_managed_by_clipper(m):
    if not clipper_installed():
        return False
    from chimerax.clipper.symmetry import SymmetryManager
    return isinstance(m.parent, SymmetryManager)

def map_associated_with_model(m, v):
    if model_managed_by_clipper(m):
        from chimerax.clipper import get_symmetry_handler
        sh = get_symmetry_handler(m)
        return (v in sh.map_mgr.all_maps)

def ensure_clipper_map_covers_selection(session, m, residues, v):
    if map_associated_with_model(m, v):
        # Only expand to cover the selection if currently in spotlight mode, otherwise we break 
        # map updating in running ISOLDE simulations.
        from chimerax.clipper import get_symmetry_handler
        sh = get_symmetry_handler(m)
        if sh.spotlight_mode:
            from chimerax.core.commands import run
            from chimerax.atomic import concise_residue_spec
            run(session, f'clipper isolate {concise_residue_spec(session, residues)}')
