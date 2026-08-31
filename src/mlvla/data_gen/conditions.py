"""Author <perturbed_root>/conditions.yaml and the 3 light scene XMLs.

Computes effective camera pos/quat via LIBERO-plus's rotate_around_y/z +
scale_distance_from_pivot so the recorded params match what the env uses at runtime.

Run in the `libero` conda env.
"""
from __future__ import annotations

import datetime
import pathlib
import re

import yaml

from mlvla import paths as _paths

_paths.add_to_sys_path()
from libero.libero.envs.problems.libero_coffee_table_manipulation import (  # noqa: E402
    rotate_around_y,
    rotate_around_z,
    scale_distance_from_pivot,
)

ROOT = pathlib.Path(_paths.get("perturbed_root"))
ROOT.mkdir(parents=True, exist_ok=True)

TEMPLATE_XML = (
    pathlib.Path(_paths.get("libero_plus_root"))
    / "libero" / "libero" / "assets" / "scenes" / "lights"
    / "tabletop_light_sync_modified_0.xml"
)
assert TEMPLATE_XML.exists(), f"Template XML missing: {TEMPLATE_XML}"

# Scene XMLs must live alongside the template so MuJoCo's relative texture
# paths resolve. We also keep a mirror copy at <perturbed_root>/<cond>/scene.xml
# for the conditions.yaml record.
SCENE_XML_DIR = TEMPLATE_XML.parent

# Default tabletop camera (from libero_tabletop_manipulation.py:306-312).
DEFAULT_POS = [0.6586131746834771, 0.0, 1.6103500240372423]
DEFAULT_QUAT = [
    0.6380177736282349,
    0.3048497438430786,
    0.30484986305236816,
    0.6380177736282349,
]

VIEW_CONDITIONS = {
    "v1_azimuth30": dict(horizon=30, vertical=0, scale_factor=1.0, endpoint_rot=0, endpoint_vertical=0),
    "v2_azimuth60": dict(horizon=60, vertical=0, scale_factor=1.0, endpoint_rot=0, endpoint_vertical=0),
    "v3_elev15_zoom125": dict(horizon=0, vertical=15, scale_factor=1.25, endpoint_rot=0, endpoint_vertical=0),
}

LIGHT_CONDITIONS = {
    "l1_warm_dim": dict(
        light1=dict(diffuse=[0.8, 0.6, 0.3], dir=[0, 0, -1], pos=[1, 1, 4], specular=[0, 0, 0], castshadow=True),
        light2=dict(diffuse=[0.4, 0.3, 0.15], dir=[0, 0, -1], pos=[-3, -3, 4], specular=[0, 0, 0], castshadow=True),
    ),
    "l2_cool_bright": dict(
        light1=dict(diffuse=[1.0, 1.1, 1.3], dir=[0, 0, -1], pos=[1, 1, 4], specular=[0, 0, 0], castshadow=False),
        light2=dict(diffuse=[0.6, 0.7, 0.9], dir=[0, 0, -1], pos=[-3, -3, 4], specular=[0, 0, 0], castshadow=True),
    ),
    "l3_directional_low": dict(
        light1=dict(diffuse=[0.5, 0.5, 0.5], dir=[0, 0, -1], pos=[2, -2, 3], specular=[0, 0, 0], castshadow=True),
        light2=dict(diffuse=[0.2, 0.2, 0.2], dir=[0, 0, -1], pos=[-1, 3, 2], specular=[0, 0, 0], castshadow=True),
    ),
}

COMBINED_CONDITIONS = {
    "c1_v1l1": ("v1_azimuth30", "l1_warm_dim"),
    "c2_v2l2": ("v2_azimuth60", "l2_cool_bright"),
    "c3_v3l3": ("v3_elev15_zoom125", "l3_directional_low"),
}


def compute_effective_camera(view):
    """Mirror libero_tabletop_manipulation.py:313-345 (vertical-y → horizon-z → scale)."""
    pos, quat = DEFAULT_POS[:], DEFAULT_QUAT[:]
    if view["vertical"] != 0:
        r = rotate_around_y(original_quat=quat, original_pos=pos, degrees=view["vertical"])
        pos, quat = r["new_pos"], r["new_quat"]
    if view["horizon"] != 0:
        r = rotate_around_z(original_quat=quat, original_pos=pos, degrees=view["horizon"])
        pos, quat = r["new_pos"], r["new_quat"]
    if view["scale_factor"] != 1.0:
        r = scale_distance_from_pivot(original_quat=quat, original_pos=pos, scale_factor=view["scale_factor"])
        pos, quat = r["new_pos"], r["new_quat"]
    return dict(pos=[float(x) for x in pos], quat=[float(x) for x in quat])


def view_block(view):
    eff = compute_effective_camera(view)
    return dict(
        horizon=view["horizon"],
        vertical=view["vertical"],
        scale_factor=view["scale_factor"],
        endpoint_rot=view["endpoint_rot"],
        endpoint_vertical=view["endpoint_vertical"],
        bddl_filename_suffix=(
            f"_view_{view['horizon']}_{view['vertical']}_"
            f"{int(round(view['scale_factor'] * 100))}_"
            f"{view['endpoint_rot']}_{view['endpoint_vertical']}_initstate_0"
        ),
        default_camera=dict(pos=DEFAULT_POS, quat=DEFAULT_QUAT),
        effective_camera=eff,
    )


def _format_attr_value(v):
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (list, tuple)):
        return " ".join(_format_attr_value(x) for x in v)
    if isinstance(v, float):
        # Match the template's "1 1 4.0" style: trim trailing zeros but keep at least one decimal.
        s = repr(v)
        return s
    return str(v)


def _render_light_tag(name, params):
    """Build a <light .../> tag with attributes in the template's order."""
    attrs = [f'name="{name}"']
    for k in ("diffuse", "dir", "pos", "specular"):
        if k in params:
            attrs.append(f'{k}="{_format_attr_value(params[k])}"')
    attrs.append('directional="false"')
    if "castshadow" in params:
        attrs.append(f'castshadow="{_format_attr_value(params["castshadow"])}"')
    return "<light " + " ".join(attrs) + "/>"


def write_scene_xml(out_path, lights):
    """Replace the first two <light .../> tags in the template with our custom tags."""
    text = TEMPLATE_XML.read_text()
    pattern = re.compile(r"<light\b[^>]*/>", re.DOTALL)
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        raise RuntimeError(f"Template XML has {len(matches)} <light/> tags, expected >=2")
    new_tags = [
        _render_light_tag("light1", lights["light1"]),
        _render_light_tag("light2", lights["light2"]),
    ]
    # Replace in reverse order so earlier offsets stay valid.
    for i in (1, 0):
        m = matches[i]
        text = text[: m.start()] + new_tags[i] + text[m.end() :]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)


def light_block(cond_id, lights):
    # Authoritative copy lives in LIBERO-plus assets so MuJoCo's relative texture paths resolve.
    canonical_name = f"tabletop_wizard_perturbed_{cond_id}.xml"
    assets_path = SCENE_XML_DIR / canonical_name
    write_scene_xml(assets_path, lights)
    # Mirror copy at <perturbed_root>/<cond>/scene.xml for conditions.yaml record.
    mirror_path = ROOT / cond_id / "scene.xml"
    write_scene_xml(mirror_path, lights)
    return dict(scene_xml=str(assets_path), scene_xml_mirror=str(mirror_path), lights=lights)


def main():
    conditions = {}

    for cid, view in VIEW_CONDITIONS.items():
        conditions[cid] = dict(type="view", view=view_block(view), light=None)

    for cid, lights in LIGHT_CONDITIONS.items():
        conditions[cid] = dict(type="light", view=None, light=light_block(cid, lights))

    for cid, (v_id, l_id) in COMBINED_CONDITIONS.items():
        v = VIEW_CONDITIONS[v_id]
        l = LIGHT_CONDITIONS[l_id]
        conditions[cid] = dict(
            type="combined",
            view=view_block(v),
            light=light_block(cid, l),
        )

    record = dict(
        generated_at=datetime.datetime.utcnow().isoformat() + "Z",
        source_dataset=_paths.get("source_dataset"),
        source_demos_per_task=50,
        libero_plus_root=_paths.get("libero_plus_root"),
        conditions=conditions,
    )
    out = ROOT / "conditions.yaml"
    out.write_text(yaml.safe_dump(record, sort_keys=False, default_flow_style=False))
    print(f"Wrote {out}")
    print(f"Wrote {len(conditions)} conditions: {list(conditions)}")
    for cid in list(LIGHT_CONDITIONS) + list(COMBINED_CONDITIONS):
        p = ROOT / cid / "scene.xml"
        print(f"  {cid}/scene.xml: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
