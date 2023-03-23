# vim: set expandtab ts=4 sw=4:

from chimerax.core.commands import (
    CmdDesc, FloatArg, Bounded, IntArg, BoolArg
)
from chimerax.map import MapArg
from chimerax.atomic import ResiduesArg


def qscore(session, residues, to_volume=None, reference_gaussian_sigma=0.6, points_per_shell=8, 
            max_shell_radius=2.0, shell_radius_step=0.1, include_hydrogens=False, deterministic=True):
    if to_volume is None:
        from chimerax.core.errors import UserError
        raise UserError("Must specify a map to compare the model to!")
    from .qscore import q_score
    residue_map, atom_scores = q_score(residues, to_volume, ref_sigma=reference_gaussian_sigma,
                                       points_per_shell=points_per_shell,
                                       max_rad=max_shell_radius, 
                                       step=shell_radius_step,
                                       include_h=include_hydrogens, 
                                       deterministic=deterministic, 
                                       logger=session.logger)
    session.logger.info(f'Overall mean Q-Score: {atom_scores.mean():.2f}')
    return residue_map, atom_scores



qscore_desc = CmdDesc(
    synopsis="Calculate the map-model Q-scores for a selection of residues",
    required=[
        ("residues", ResiduesArg),
    ],
    keyword=[
        ("to_volume", MapArg),
        ("reference_gaussian_sigma", Bounded(FloatArg, min=0.01)),
        ("points_per_shell", Bounded(IntArg, min=1)),
        ("max_shell_radius", Bounded(FloatArg, min=0.6, max=2.5)),
        ("shell_radius_step", Bounded(FloatArg, min=0.05, max=0.5)),
        ("include_hydrogens", BoolArg),
        ("deterministic", BoolArg)
    ]
    )
