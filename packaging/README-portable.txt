GlassTranslate {version} - portable bundle
==========================================

Live screen translation for Windows.  Nothing is installed, nothing is
downloaded, and nothing ever leaves this machine.


WHAT YOU DOWNLOADED
-------------------

  GlassTranslate-{version}-portable-win64.zip   the program (about 3 GB)
  GlassTranslate-{version}-models-win64.zip     the models (about {models_zip_gb} GB)

Both zips contain the same top-level folder, GlassTranslate-{version}.


HOW TO START
------------

  1. Extract BOTH zips into the SAME folder.  You end up with one folder,
     GlassTranslate-{version}, holding GlassTranslate.exe, renderer\ and
     models\.
  2. Double-click GlassTranslate.exe.

The first launch is slower than later ones: Windows Defender scans about 3 GB
of new files once.  Give it a minute.  The same applies to the quality
renderer: its first job after unpacking can take up to ten minutes while
Defender reads the 7 GB checkpoint for the first time; from the second launch
on it is ready in about a minute.  If SmartScreen asks, the bundle is
unsigned - use "More info" then "Run anyway", or delete the folder.


WHERE YOUR DATA LIVES
---------------------

A file called portable.txt sits next to GlassTranslate.exe.  While it is
there, everything stays inside this one folder:

  models\    the OCR, translation and quality-renderer models
  config\    settings (config.json) and your API keys (secrets.json)
  logs\      the application and sidecar logs

Nothing is written to AppData, the registry or anywhere else.  You can move,
rename or copy the whole folder - including onto a USB drive - and it keeps
working.  Delete portable.txt and the app reverts to the normal per-user
location under %LOCALAPPDATA%\GlassTranslate.

config\secrets.json stores your API key in plain text - delete it before
copying or sharing this folder.


THE QUALITY RENDERER (OPTIONAL)
-------------------------------

The renderer\ folder redraws artwork behind removed text instead of flat-
filling it.  It needs:

  - an NVIDIA GPU with about 16 GB of VRAM,
  - a recent NVIDIA driver (the CUDA 13 runtime itself is inside renderer\,
    so there is nothing else to install).

Turn it on under Engines -> Quality renderer.  Without a suitable GPU leave it
off: the app works normally and uses its built-in quick fill instead.  You can
also delete the whole renderer\ folder if you never want it.


OFFLINE
-------

No network connection is needed or used at any point.  Every model ships in
the models zip, already verified against its published SHA-256; the file
models\MANIFEST.json lists every shipped file with its size and digest, the
translation packs it contains, and the language pairs those packs do NOT
cover.

{translation_packs}


LICENCES
--------

licenses\LICENSES.md lists every component of the program, the sidecar and
the models with the licence it is used under; the full licence texts are in
the same folder, one directory per package.


REMOVING IT
-----------

Delete the GlassTranslate-{version} folder.  That is all - there is no
installer, no service, no registry key and no leftover data.


Built {date}.
