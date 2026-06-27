@echo off
setlocal EnableExtensions
REM ChimeraX launches go through run_chimerax.bat so each worktree/lane installs
REM into its own isolated ChimeraX user directory (see run_chimerax.bat). The
REM "release" token (stable vs daily install) is forwarded to it; "clean" and
REM "app-install" select the action. This bundle has C++ sources but no vcvars
REM bootstrap here -- run from a VS2017 (v141) vcvars shell, as before.
set "RELEASE_TOK="
set DO_CLEAN=
set DO_INSTALL=
for %%A in (%*) do (
    if /i "%%A"=="release"     set "RELEASE_TOK=release"
    if /i "%%A"=="clean"       set DO_CLEAN=1
    if /i "%%A"=="app-install" set DO_INSTALL=1
)

if defined DO_CLEAN   call "%~dp0run_chimerax.bat" %RELEASE_TOK% --nogui --safemode --exit --cmd "devel clean ."
if defined DO_INSTALL call "%~dp0run_chimerax.bat" %RELEASE_TOK% --nogui --safemode --exit --cmd "devel install ."

endlocal
