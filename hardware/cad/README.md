# CAD

Mechanical design files and exported fabrication assets.

```text
hardware/cad/
  fusion360_source/   Fusion 360 source archives, such as .f3d or .f3z
  step/               editable CAD exchange files for solid parts
  stl/                3D-printable mesh exports
  export_inbox/       temporary manual export drop folder
```

For each printable solid part, publish both `.step` and `.stl` when possible.
Mesh-only face-cover parts can be published as `.stl` unless they are converted
to solid BRep bodies in Fusion 360.

See `EXPORT_MANIFEST.md` for the exact printable parts list.
