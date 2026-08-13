# Fusion 360 Export Tools

This folder contains helper scripts for exporting selected parts from the
Fusion 360 design.

## Export Workflow

1. Open `headlike robotic eye` in Fusion 360.
2. Save a full source archive manually:
   - `File > Export > Fusion 360 Archive (*.f3z)`
   - Save as `hardware/cad/fusion360_source/headlike_robotic_eye_assembly.f3z`.
3. In Fusion 360, open:
   - `Utilities > Scripts and Add-Ins`.
4. Create or add a Python script and copy the contents of
   `export_headlike_robotic_eye_parts.py`.
5. Run the script while the design is open.
6. Check the generated files in:
   - `hardware/cad/step/`
   - `hardware/cad/stl/`

If a mesh face-cover body cannot be exported by the script, manually right-click
that mesh body and export it as STL using the filenames in
`hardware/cad/EXPORT_MANIFEST.md`.
