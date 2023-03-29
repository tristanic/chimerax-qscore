.. _qscore:

qscore
======

Syntax: qscore *residues* **toVolume** *a density map specifier* 
[**useGui** *true or false*] [**referenceGaussianSigma** *float*] 
[**pointsPerShell** *integer*] [**maxShellRadius** *float*] 
[**shellRadiusStep** *float*] [**logDetails** *true or false*] 
[**outputFile** *a file name*]

The map-model Q-score, as described in `Pintille *et al.* (2020)`__,
aims to estimate the "resolvability" of each non-hydrogen atom in a given
map by comparing the surrounding density dominated by that atom to what 
you would expect to see in a high-resolution map. It does this by defining
a series of spherical shells around the atom at intervals of *shellRadiusStep*
up to *maxShellRadius*, and attempting to find *pointsPerShell* points in each 
shell that are closer to the test atom than any other atom in the model. 
The interpolated map values measured at each of these point are compared to 
the expected values derived from an ideal Gaussian with a sigma value of
*referenceGaussianSigma*. By default the command will launch a GUI browser 
to explore the full results, and will report the overall average 
Q-score to the log. Setting *logDetails* to *true* will print a more 
comprehensive residue-by-residue summary to the log, and providing 
*outputFile* will save the same details in CSV format.

__ https://www.nature.com/articles/s41592-020-0731-1

**residues** (required)
-----------------------

A selection string defining a set of residues from a **single** model.
Note that if *useGui* is *true* the calculation will be performed on the 
entire model; otherwise only those residues in the selection will be 
analysed.

**toVolume** (required)
-----------------------

A selection string defining a single volumetric map. The model should 
already be fitted into the map.

**useGui**
----------

If *true* (default), the Q-Score GUI will be launched allowing you to 
explore the results via an interactive plot.

**referenceGaussianSigma**
--------------------------

A floating point value greater than zero. The default value (0.6) 
corresponds approximately to a resolution of 1.3 Angstroms, and is 
the value used for Q-score validation when depositing to the wwPDB.
In most cases this should not be changed.

**pointsPerShell**
------------------

An integer greater than zero (default 8). The number of test points 
to use in each shell. Smaller numbers give faster calculations but 
noisier individual scores.

**maxShellRadius**
------------------

A floating point value greater than zero (default 2.0). The largest 
allowable radius of a test shell. Should not be much larger nor smaller 
than the radius of an atom.

**shellRadiusStep**
-------------------

A floating point value greater than zero (default 0.1). A larger step 
size gives faster calculations at the expense of a noisier result.

**logDetails**
--------------

False by default. If true, a residue-by-residue summary similar to the 
below will be printed to the log::

    Chain	Number	Name	Qavg	Qworst	Qbb	Qsc
    -------------------------------------------------------
    A   	281  	GLY 	0.582	0.360	0.582	N/A
    A   	282  	GLY 	0.445	0.367	0.445	N/A
    A   	283  	PHE 	0.526	-0.119	0.367	0.617

Here *Qavg* give the overall average score for the residue; *Qworst*
gives the score for the worst atom in that residue; *Qbb* gives the 
average score for backbone atoms, and *Qsc* gives the average score 
for sidechain atoms. A value of N/A means no score calculation was 
possible. This is usually because no atoms of a given category were 
found (e.g. for GLY in the example above, which has no sidechain). 
Any residue with at least one atom falling outside the map extent 
will have a score of N/A in all categories.

**outputFile**
---------------

If given, the same details as above will be written in comma-separated 
text (CSV) format to a file of that name.