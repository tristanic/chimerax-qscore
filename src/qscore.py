def unit_sphere_vertices(num_vertices):
    import numpy
    from chimerax.surface.shapes import sphere_geometry2
    sphere_vertices = sphere_geometry2(2*num_vertices-4)[0] # 128 points evenly distributed around a unit sphere centred on (0,0,0)
    return sphere_vertices

def min_max_d(v):
    m = v.data.full_matrix()
    mean, sd, _ = v.mean_sd_rms()
    max_d = min (mean+sd*10, m.max())
    min_d = max (mean-sd, m.min())
    return min_d, max_d

_numpy_print_options = {
    'precision':3, 
    'suppress':True
    }

RANDOM_SEED=1985

def q_score(residues, volume, 
            ref_sigma=0.6, points_per_shell = 8, max_rad = 2.0, step=0.1, num_test_points=128, 
            clustering_iterations=5, include_h = False, debug = False, draw_points=False, 
            logger=None, log_interval=1000, log_details = False, output_file = None,assign_attr=False,
            randomize_shell_points=True, random_seed=RANDOM_SEED,):
    '''
    Implementation of the map-model Q-score as described in Pintille et al. (2020): https://www.nature.com/articles/s41592-020-0731-1.

    If the model is well-fitted to the map, the Q-score is essentially an atom-by-atom estimate of "resolvability"
    (i.e. how much the map tells us about the true position of the atom). If the model is *not* well-fitted, then 
    low Q-scores are good pointers to possible problem regions. In practice, of course, usually we have a mixture of 
    both situations.

    This version of the algorithm has a few minor modifications compared to the original implementation aimed at 
    improving overall speed. As a result the scores it returns are not identical to the original, typically differing by 
    +/- 0.04 in individual atom scores and +/- 0.02 in residue averages. This difference can be explained by the different 
    choice of test points, and reflects the underlying sampling uncertainty in the method. 
    
    This implementation works as follows:

    - For each atom, define a set of shells in steps of `step` angstroms out to `max_rad` angstroms.
    - For each shell, try to find at least `points_per_shell` probe points closer to the test atom than 
      any other atom:

      - For radii smaller than about half a bond length, first try a set of `points_per_shell`
        points evenly spread around the spherical surface. 
      - For larger radii (or if this quick approach fails to find enough points on smaller radii), 
        start with `num_test_points` evenly spread on the sphere surface, remove points closer to other atoms 
        than the test atom. If more than `points_per_shell` points remain, perform up to `clustering_iterations`
        rounds of k-means clustering to find `points_per_shell` clusters, and choose the point nearest to the 
        centroid of each cluster. If <= `points_per_shell` points remain, just do the best with what we have.
        By default, the "seed" centroids for each cluster are chosen pseudo-randomly from the input points.
        Using the same `random_seed` will give the same result each time; varying `random_seed` over multiple 
        runs can be used to give an idea of the underlying uncertainty in the algorithm. If `randomize_shell_points`
        is False, the seed centroids will instead be the closest point (in spherical coordinates) to each of 
        `points_per_shell` evenly-spaced points on a unit sphere. While this may intuitively seem preferable,
        in practice for tightly-packed atoms it leads to oversampling of the "junctions" with other atoms, and 
        undersampling of the unhindered space. 

    - Interpolate the local density value at each shell point (including at the centre of the atom itself),
      and calculate the Q-score as the normalised-about-the-mean cross-correlation between the measured values 
      and an ideal Gaussian with `ref_sigma` fall-off (the default of 0.6 corresponds to approximately the 
      expected fall-off in a 1.3 Angstrom map).

    On a typical machine, run time is on the order of 1 ms/atom (i.e. from 10s to a few minutes for typical 
    cryo-EM models). If the ChimeraX logger (`session.logger`) is passed in as the `logger` argument, the approximate
    remaining time will be printed to the status bar every `log_interval` atoms.

    The `debug` and `draw_points` arguments are meant primarily for debugging, and should only be used for 
    small selections at a time (`debug` is *very*  verbose, and `draw_points` leads to the drawing of approx.
    `max_rad`/`step`*`points_per_shell` ~ 160 points per atom).

    If `assign_attr` is True, then each selected atom will have the qscore attribute assigned.

    Returns:

    - a {Residue: (mean_q_score, worst_atom_score)} dict
    - a numpy array with the Q-scores for all atoms in order of `residues.atoms` 
    '''
    from chimerax.geometry import find_close_points, find_closest_points, Places
    from chimerax.atomic import Residues
    import numpy
    from math import floor
    if len(residues.unique_structures) != 1:
        from chimerax.core.errors import UserError
        raise UserError('All residues must be from the same model!')
    m = residues.unique_structures[0]
    session = residues.unique_structures[0].session
    from .clipper_compat import ensure_clipper_map_covers_selection, return_clipper_model_to_spotlight_mode
    was_spotlight = ensure_clipper_map_covers_selection(session, m, residues, volume)
    from . import _kmeans
    matrix, xform = volume.matrix_and_transform(None, 'all', (1,1,1))
    from chimerax.map_data import interpolate_volume_data
    min_d, max_d = min_max_d(volume)
    pps_vertices = unit_sphere_vertices(points_per_shell)
    ref_sphere_vertices = unit_sphere_vertices(num_test_points)

    a = max_d - min_d
    b = min_d

    radii = numpy.arange(0, max_rad+step/2, step)
    reference_gaussian = a * numpy.exp( -0.5 * (radii/ref_sigma)**2) + b

    query_atoms = residues.atoms
    all_atoms = residues.unique_structures[0].atoms
    if not include_h:
        query_atoms = query_atoms[query_atoms.element_names != 'H']
        all_atoms = all_atoms[all_atoms.element_names != 'H']

    # register the qscore attribute in case this is needed
    if assign_attr:
        from chimerax.atomic import Atom
        Atom.register_attr(session, 'qscore', 'qscore', attr_type=float, can_return_none=True)

    if draw_points:
        from chimerax.core.models import Model, Drawing
        from chimerax.surface.shapes import sphere_geometry2
        from chimerax.core.colors import random_colors
        d = Drawing('shell points')
        v, n, t = sphere_geometry2(24)
        v *= (0.05)
        d.set_geometry(v, n, t)
        dm = Model('shell points', session)
        dm.add_drawing(d)
        positions = []
        base_colors = random_colors(len(query_atoms))
        colors = []

    all_coords = all_atoms.scene_coords
    query_coords = query_atoms.scene_coords

    r0_vals, oob = volume.interpolated_values(query_coords, out_of_bounds_list=True)
    oob_residues = Residues()
    if len(oob):
        oob_residues = query_atoms[oob].unique_residues
        from chimerax.atomic import concise_residue_spec
        session.logger.warning(f'WARNING: the following residues in #{m.id_string} have atoms outside the bounds of volume #{volume.id_string} '
                            f'and will be excluded from the Q-score calculation:\n{concise_residue_spec(session, oob_residues)}')
        mask = numpy.logical_not(numpy.in1d(query_atoms.residues, oob_residues))
        residues = residues[numpy.logical_not(numpy.in1d(residues, oob_residues))]
        query_atoms = query_atoms[mask]
        query_coords = query_atoms.scene_coords
        r0_vals = r0_vals[mask]
    
    num_shells = int(floor(max_rad/step))

    if logger is not None:
        from time import time
        start_time = time()


    q_scores = []
    for i,a in enumerate(query_atoms):
        if draw_points:
            color = base_colors[i]
        if logger is not None and i!= 0 and i%log_interval==0:
            current_time = time()
            elapsed = current_time - start_time
            estimated_total = len(query_atoms)/i * elapsed
            logger.status(f'Estimated time remaining: {estimated_total-elapsed:.1f} seconds')
        a_coord = a.scene_coord
        _, nearby_i = find_close_points([a_coord], all_coords, max_rad*3)
        nearby_a = all_atoms[nearby_i]
        ai = nearby_a.index(a)
        nearby_coords = nearby_a.scene_coords
        shell_rad = step
        local_d_vals = {}
        j = 1
        while shell_rad < max_rad+step/2:
            local_pps = (pps_vertices*shell_rad) + a_coord
            if shell_rad < 0.7: # about half a C-C bond length
                # Try the quick way first (should succeed for almost all cases unless geometry is seriously wonky)
                i1, i2, near1 = find_closest_points(local_pps, nearby_coords, shell_rad*1.5)
                closest = near1
                candidates = i1[closest==ai]
                if len(candidates)==points_per_shell:
                    d_vals = interpolate_volume_data(local_pps, xform, matrix, 'linear')[0]
                    if draw_points:
                        positions.append(local_pps)
                        colors.append(numpy.array([color]*points_per_shell))
                    local_d_vals[j] = d_vals
                    shell_rad += step
                    j+=1
                    continue
            
            local_sphere = (ref_sphere_vertices*shell_rad) + a_coord
            i1, i2, near1 = find_closest_points(local_sphere, nearby_coords, shell_rad*1.5)
            closest = near1
            candidates = i1[closest==ai]
            if len(candidates) == 0:
                if debug:
                    id_string = f'#{a.residue.structure.id_string}/{a.residue.chain_id}:{a.residue.number}@{a.name}'
                    print(f'WARNING: no shell points found for atom {id_string} at radius {shell_rad:.1f}!')
            elif len(candidates) < points_per_shell:
                points = local_sphere[candidates]
                d_vals = interpolate_volume_data(points, xform, matrix, 'linear')[0]
                local_d_vals[j] = d_vals
            else:
                points = local_sphere[candidates]
                if not randomize_shell_points:
                    labels, closest = _kmeans.spherical_k_means_defined(points, a_coord, points_per_shell, local_pps, clustering_iterations)
                else:
                    labels, closest = _kmeans.spherical_k_means_random(points, a_coord, points_per_shell, clustering_iterations, random_seed+j)
                if debug:
                    with numpy.printoptions(**_numpy_print_options):
                        print(f'Points: {points}')
                        print(f'Labels: {labels}')
                        print(f'Closest indices: {closest}')
                points = points[closest]
                d_vals = interpolate_volume_data(points, xform, matrix, 'linear')[0]
                local_d_vals[j] = d_vals
            if draw_points:
                if len(candidates) != 0:
                    positions.append(points)
                    colors.append(numpy.array([color]*len(points)))
            shell_rad += step
            j+=1
        from chimerax.map_fit import overlap_and_correlation
        measured = numpy.concatenate([[r0_vals[i]]*8, *local_d_vals.values()])
        ref = numpy.concatenate([[reference_gaussian[0]]*8, *[[reference_gaussian[j]] * len(vals) for j,vals in local_d_vals.items()]])
        q = overlap_and_correlation(ref, measured)[2]
        if debug:
            with numpy.printoptions(**_numpy_print_options):
                print(f'Measured: {measured}')
                print(f'Ref: {ref}')
                print(f'q: {q:.3f}')
        q_scores.append(q)
        # assign the attribute if requested
        if assign_attr:
            a.qscore = q

    if draw_points:
        positions = numpy.concatenate(positions)
        colors = numpy.concatenate(colors)
        shift_and_scale = numpy.ones((len(positions),4), dtype=numpy.float32)
        shift_and_scale[:,:3] = positions
        positions = Places(shift_and_scale=shift_and_scale)
        d.positions = positions
        d.colors = colors
        session.models.add([dm])
        
    q_scores = numpy.array(q_scores)
    residue_scores = {}
    for r in residues:
        indices = query_atoms.indices(r.atoms)
        indices = indices[indices!=-1]
        if len(indices) > 0:
            rscores = q_scores[indices]
            residue_scores[r] = (rscores.mean(), rscores.min())
    
    if was_spotlight:
        return_clipper_model_to_spotlight_mode(session, m)
    if logger is not None:
        logger.status('')
    if log_details or output_file:
        report_results(residue_scores, query_atoms, q_scores, oob_residues, log=log_details, filename=output_file)
    return residue_scores, (query_atoms, q_scores)

def report_results(residue_map, query_atoms, atom_scores, out_of_bounds, log=False, filename=None):
    log_sep = '\t'
    file_sep = ','
    from chimerax.atomic import Residues, concatenate
    residues = concatenate([Residues(residue_map.keys()), out_of_bounds])
    residues = Residues(sorted(residues, key=lambda r:(r.chain_id, r.number, r.insertion_code)))
    logger = residues[0].session.logger
    if filename is None:
        out = None
    else:
        out = open(filename, 'wt')
    if log:
        sep=log_sep
        header = f'<pre>Chain{sep}Number{sep}Name{sep}Qavg{sep}Qworst{sep}Qbb{sep}Qsc</pre>'
        logger.info(header, is_html=True, add_newline=False)
        logger.info(f'<pre>{"-"*(len(header.expandtabs())-10)}</pre>', is_html=True, add_newline=False)
    if filename is not None:
        sep = file_sep
        print(f'Chain{sep}Number{sep}Name{sep}Qavg{sep}Qworst{sep}Qbackbone{sep}Qsidechain', file=out)
    for r in residues:
        scores = residue_map.get(r, None)
        if scores is None:
            if log:
                sep = log_sep
                logger.info(f'<pre>{r.chain_id}{sep}{r.number}{r.insertion_code}{sep}{r.name}{sep}"N/A"{sep}"N\A"{sep}"N/A"{sep}"N/A"</pre>', is_html=True, add_newline=False)
            if filename is not None:
                sep = file_sep
                print(f'{r.chain_id}{sep}{r.number}{r.insertion_code}{sep}{r.name}{sep}"N/A"{sep}"N\A"{sep}"N/A"{sep}"N/A"', file=out)
        qavg, qworst = scores
        backbone_atoms = r.atoms[r.atoms.is_backbones()]
        if len(backbone_atoms):
            bi = query_atoms.indices(backbone_atoms)
            qbackbone = f'{atom_scores[bi].mean():.3f}'
        else:
            qbackbone = 'N/A'
        sidechain_atoms = r.atoms[r.atoms.is_side_onlys]
        if len(sidechain_atoms):
            si = query_atoms.indices(sidechain_atoms)
            qsidechain = f'{atom_scores[si].mean():.3f}'
        else:
            qsidechain = 'N/A'

        if log:
            sep = log_sep
            logger.info(f'<pre>{r.chain_id}{sep}{r.number}{r.insertion_code}{sep}{r.name}{sep}{qavg:.3f}{sep}{qworst:.3f}{sep}{qbackbone}{sep}{qsidechain}</pre>', is_html=True, add_newline=False)
        if filename is not None:
            sep = file_sep
            print(f'{r.chain_id}{sep}{r.number}{r.insertion_code}{sep}{r.name}{sep}{qavg:.3f}{sep}{qworst:.3f}{sep}{qbackbone}{sep}{qsidechain}', file=out)
    if out is not None:
        out.close()
    if log:
        logger.info('')


def test_q_score(session):
    from chimerax.core.commands import run
    m = run(session, 'open 6out')[0]
    v = run(session, 'open 20205 from emdb')[0]
    residue_map, atom_scores = q_score(m.residues, v)
    from matplotlib import pyplot as plt
    fig = plt.figure()
    plots = fig.subplots(len(m.chains))
    for i,c in enumerate(m.chains):
        plot = plots[i]
        residues = c.residues
        x = []
        y = []
        x = [r.number for r in residues if r is not None]
        y = [residue_map[r][0] for r in residues if r is not None]
        plot.plot(x, y)
        plot.set_xlabel('Residue number')
        plot.set_ylabel('Q Score')
        plot.set_title(f'Chain {c.chain_id}')
    fig.tight_layout()
    fig.show()












    





