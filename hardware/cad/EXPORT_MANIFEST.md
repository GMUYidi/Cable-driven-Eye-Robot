# CAD Export Manifest

This manifest defines the CAD files that should be exported from the Fusion 360
design `headlike robotic eye` for public release and fabrication.

## Required Printable Parts

| Output name | Fusion 360 source name | Quantity | Export formats |
| --- | --- | ---: | --- |
| `eyeball_r1` | `eyeball_r1` | 1 | STEP, STL |
| `eyeball_r2` | `eyeball_r2` | 1 | STEP, STL |
| `front_eye_support_right` | `front eye support of right eye` | 1 | STEP, STL |
| `main_base_right_eye` | `Main base of right eye (1)` | 1 | STEP, STL |
| `support4` | `support4` | 1 | STEP, STL |
| `main_base_left_eye` | `Main base of left eye` | 1 | STEP, STL |
| `eyeball_l1` | `eyeball_l1` | 1 | STEP, STL |
| `eyeball_l2` | `eyeball_l2` | 1 | STEP, STL |
| `front_eye_support_left` | `front eye support of left eye` | 1 | STEP, STL |
| `connection` | `connection` | 1 | STEP, STL |
| `servo_spool_1` | `servo spool 1` | 12 | STEP, STL |
| `servo_spool_2` | `servo spool 2` | 12 | STEP, STL |
| `face_cover_upper` | `eyeglass` -> `MeshBody 1` / `网格实体1` | 1 | STL |
| `face_cover_lower` | `topmouth` -> `MeshBody 1` / `网格实体1` | 1 | STL |

## Optional Assembly Exports

Export these manually from Fusion 360 if possible:

- `headlike_robotic_eye_assembly.f3z`
- `headlike_robotic_eye_assembly.step`

Place them in:

```text
hardware/cad/fusion360_source/
hardware/cad/step/
```

## Notes

- STEP is the preferred editable CAD exchange format for solid parts.
- STL is the preferred fabrication format for 3D printing.
- Mesh bodies generally export cleanly as STL. If a mesh body must also be
  edited as CAD, convert it to a solid BRep in Fusion 360 first, then export
  STEP.
