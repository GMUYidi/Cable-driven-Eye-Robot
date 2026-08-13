# Fusion 360 script: export selected printable parts from "headlike robotic eye".
#
# Run inside Fusion 360:
#   Utilities > Scripts and Add-Ins > Python script > Run
#
# The script searches the active design by body/component name and exports the
# selected parts into this repository's hardware/cad/step and hardware/cad/stl
# folders.

import os
import re
import traceback

import adsk.core
import adsk.fusion


REPO_CAD_DIR = os.path.join(
    os.environ.get("USERPROFILE", r"C:\Users\huang"),
    "OneDrive",
    "\u684c\u9762",
    "robotic eye github",
    "hardware",
    "cad",
)
STEP_DIR = os.path.join(REPO_CAD_DIR, "step")
STL_DIR = os.path.join(REPO_CAD_DIR, "stl")
LOG_FILE = os.path.join(REPO_CAD_DIR, "export_log.txt")


PARTS = [
    {
        "output": "eyeball_r1",
        "aliases": ["eyeball_r1"],
        "quantity": 1,
        "formats": ["step", "stl"],
    },
    {
        "output": "eyeball_r2",
        "aliases": ["eyeball_r2"],
        "quantity": 1,
        "formats": ["step", "stl"],
    },
    {
        "output": "front_eye_support_right",
        "aliases": ["front eye support of right eye", "front eye support of right"],
        "quantity": 1,
        "formats": ["step", "stl"],
    },
    {
        "output": "main_base_right_eye",
        "aliases": ["main base of right eye", "main base of right eye (1)"],
        "quantity": 1,
        "formats": ["step", "stl"],
    },
    {
        "output": "support4",
        "aliases": ["support4"],
        "quantity": 1,
        "formats": ["step", "stl"],
    },
    {
        "output": "main_base_left_eye",
        "aliases": ["main base of left eye"],
        "quantity": 1,
        "formats": ["step", "stl"],
    },
    {
        "output": "eyeball_l1",
        "aliases": ["eyeball_l1"],
        "quantity": 1,
        "formats": ["step", "stl"],
    },
    {
        "output": "eyeball_l2",
        "aliases": ["eyeball_l2"],
        "quantity": 1,
        "formats": ["step", "stl"],
    },
    {
        "output": "front_eye_support_left",
        "aliases": ["front eye support of left eye", "front eye support of left"],
        "quantity": 1,
        "formats": ["step", "stl"],
    },
    {
        "output": "connection",
        "aliases": ["connection"],
        "quantity": 1,
        "formats": ["step", "stl"],
    },
    {
        "output": "servo_spool_1",
        "aliases": ["servo spool 1", "servo_spool_1"],
        "quantity": 12,
        "formats": ["step", "stl"],
    },
    {
        "output": "servo_spool_2",
        "aliases": ["servo spool 2", "servo_spool_2"],
        "quantity": 12,
        "formats": ["step", "stl"],
    },
    {
        "output": "face_cover_upper",
        "aliases": ["eyeglass", "eye glass", "face cover upper"],
        "mesh_parent_aliases": ["eyeglass"],
        "mesh_body_aliases": [],
        "quantity": 1,
        "formats": ["stl"],
    },
    {
        "output": "face_cover_lower",
        "aliases": ["topmouth", "top mouth", "face cover lower"],
        "mesh_parent_aliases": ["topmouth"],
        "mesh_body_aliases": [],
        "quantity": 1,
        "formats": ["stl"],
    },
]


def normalize(value):
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", value or "").lower()


def matches(name, aliases):
    normalized_name = normalize(name)
    for alias in aliases:
        normalized_alias = normalize(alias)
        if normalized_alias and normalized_alias in normalized_name:
            return True
    return False


def ensure_dirs():
    os.makedirs(STEP_DIR, exist_ok=True)
    os.makedirs(STL_DIR, exist_ok=True)


def collect_brep_bodies(component, path="", seen=None):
    if seen is None:
        seen = set()
    rows = []
    token = component.entityToken
    if token in seen:
        return rows
    seen.add(token)

    current_path = f"{path}/{component.name}" if path else component.name
    for body in component.bRepBodies:
        rows.append(
            {
                "body": body,
                "body_name": body.name,
                "path": current_path,
                "component": component,
                "kind": "brep",
            }
        )

    for occurrence in component.occurrences:
        occurrence_path = f"{current_path}/{occurrence.name}"
        try:
            for body in occurrence.bRepBodies:
                rows.append(
                    {
                        "body": body,
                        "body_name": body.name,
                        "path": occurrence_path,
                        "component": occurrence.component,
                        "kind": "brep",
                    }
                )
        except Exception:
            pass
        rows.extend(collect_brep_bodies(occurrence.component, occurrence_path, seen))
    return rows


def collect_mesh_bodies(component, path="", seen=None):
    if seen is None:
        seen = set()
    rows = []
    token = component.entityToken
    if token in seen:
        return rows
    seen.add(token)

    current_path = f"{path}/{component.name}" if path else component.name
    for body in component.meshBodies:
        rows.append(
            {
                "body": body,
                "body_name": body.name,
                "path": current_path,
                "component": component,
                "kind": "mesh",
            }
        )

    for occurrence in component.occurrences:
        occurrence_path = f"{current_path}/{occurrence.name}"
        try:
            for body in occurrence.meshBodies:
                rows.append(
                    {
                        "body": body,
                        "body_name": body.name,
                        "path": occurrence_path,
                        "component": occurrence.component,
                        "kind": "mesh",
                    }
                )
        except Exception:
            pass
        rows.extend(collect_mesh_bodies(occurrence.component, occurrence_path, seen))
    return rows


def find_solid(part, brep_rows):
    for row in brep_rows:
        if matches(row["body_name"], part["aliases"]) or matches(row["path"], part["aliases"]):
            return row
    return None


def find_mesh(part, mesh_rows):
    parent_aliases = part.get("mesh_parent_aliases", part["aliases"])
    body_aliases = part.get("mesh_body_aliases", [])
    for row in mesh_rows:
        parent_match = matches(row["path"], parent_aliases)
        body_match = not body_aliases or matches(row["body_name"], body_aliases)
        if parent_match and body_match:
            return row
    for row in mesh_rows:
        if matches(row["path"], part["aliases"]) or matches(row["body_name"], part["aliases"]):
            return row
    return None


def export_step_from_body(root_component, export_manager, body, output_name, filename):
    transform = adsk.core.Matrix3D.create()
    temp_occurrence = root_component.occurrences.addNewComponent(transform)
    temp_occurrence.component.name = f"_export_tmp_{output_name}"
    try:
        body.copyToComponent(temp_occurrence)
        options = export_manager.createSTEPExportOptions(filename, temp_occurrence.component)
        export_manager.execute(options)
    finally:
        try:
            temp_occurrence.deleteMe()
        except Exception:
            pass


def export_stl(export_manager, geometry, filename):
    options = export_manager.createSTLExportOptions(geometry)
    options.filename = filename
    options.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
    export_manager.execute(options)


def export_stl_from_body(root_component, export_manager, body, output_name, filename):
    transform = adsk.core.Matrix3D.create()
    temp_occurrence = root_component.occurrences.addNewComponent(transform)
    temp_occurrence.component.name = f"_export_tmp_stl_{output_name}"
    try:
        body.copyToComponent(temp_occurrence)
        options = export_manager.createSTLExportOptions(temp_occurrence.component)
        options.filename = filename
        options.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
        export_manager.execute(options)
    finally:
        try:
            temp_occurrence.deleteMe()
        except Exception:
            pass


def cleanup_expected_outputs():
    for part in PARTS:
        for fmt in part["formats"]:
            folder = STEP_DIR if fmt == "step" else STL_DIR
            filename = os.path.join(folder, f"{part['output']}.{fmt}")
            if os.path.exists(filename):
                os.remove(filename)


def checked_export(export_func, filename):
    if os.path.exists(filename):
        os.remove(filename)
    export_func()
    if not os.path.exists(filename):
        raise RuntimeError(f"Fusion reported success, but file was not created: {filename}")


def checked_stl_export(root_component, export_manager, row, output_name, filename):
    errors = []
    body = row["body"]

    try:
        checked_export(lambda: export_stl(export_manager, body, filename), filename)
        return
    except Exception as exc:
        errors.append(f"direct body export failed: {exc}")

    if row.get("kind") == "brep":
        try:
            checked_export(
                lambda: export_stl_from_body(root_component, export_manager, body, output_name, filename),
                filename,
            )
            return
        except Exception as exc:
            errors.append(f"temporary component export failed: {exc}")

    raise RuntimeError("; ".join(errors))


def write_log(brep_rows, mesh_rows, found_rows, missing, warnings, exported):
    lines = []
    lines.append("CAD export log")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Matched parts:")
    for output, row in found_rows:
        lines.append(f"- {output}: body='{row['body_name']}' path='{row['path']}'")
    lines.append("")
    lines.append("Missing parts:")
    for item in missing:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Warnings:")
    for item in warnings:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Exported files:")
    for item in exported:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("All BRep bodies seen by the script:")
    for row in brep_rows:
        lines.append(f"- body='{row['body_name']}' path='{row['path']}'")
    lines.append("")
    lines.append("All mesh bodies seen by the script:")
    for row in mesh_rows:
        lines.append(f"- body='{row['body_name']}' path='{row['path']}'")

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("\n".join(lines))


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox("No active Fusion design is open.")
            return

        ensure_dirs()
        cleanup_expected_outputs()
        export_manager = design.exportManager
        root = design.rootComponent
        brep_rows = collect_brep_bodies(root)
        mesh_rows = collect_mesh_bodies(root)

        missing = []
        exported = []
        warnings = []
        found_rows = []

        for part in PARTS:
            target = None
            if part["formats"] == ["stl"]:
                target = find_mesh(part, mesh_rows) or find_solid(part, brep_rows)
            else:
                target = find_solid(part, brep_rows)
            if target is None:
                missing.append(part["output"])
                continue

            found_rows.append((part["output"], target))
            body = target["body"]
            if "step" in part["formats"]:
                step_file = os.path.join(STEP_DIR, f"{part['output']}.step")
                try:
                    checked_export(
                        lambda: export_step_from_body(root, export_manager, body, part["output"], step_file),
                        step_file,
                    )
                    exported.append(step_file)
                except Exception as exc:
                    warnings.append(f"STEP failed for {part['output']}: {exc}")

            if "stl" in part["formats"]:
                stl_file = os.path.join(STL_DIR, f"{part['output']}.stl")
                try:
                    checked_stl_export(root, export_manager, target, part["output"], stl_file)
                    exported.append(stl_file)
                except Exception as exc:
                    warnings.append(f"STL failed for {part['output']}: {exc}")

        write_log(brep_rows, mesh_rows, found_rows, missing, warnings, exported)

        summary = [
            f"Exported {len(exported)} files.",
            f"STEP folder: {STEP_DIR}",
            f"STL folder: {STL_DIR}",
            f"Log file: {LOG_FILE}",
        ]
        if missing:
            summary.append("\nMissing parts:\n- " + "\n- ".join(missing))
        if warnings:
            summary.append("\nWarnings:\n- " + "\n- ".join(warnings))

        ui.messageBox("\n".join(summary))

    except Exception:
        if ui:
            ui.messageBox("CAD export failed:\n\n" + traceback.format_exc())


def stop(context):
    pass
