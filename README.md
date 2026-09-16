# Sync-demo

Sync-demo is a local, English-language web application for manually annotating a surgical instrument and fitting a constrained, state-driven 3D-to-2D visual registration against a left endoscope video.

> **Research use only.** This is an exploratory visual-registration tool, not a validated hand-eye calibration and not a transform for robot control. A low reprojection error does not establish physical accuracy.

## Model

The robot observation supplies a 6D tool pose for every video frame:

- position in metres;
- quaternion in `qx, qy, qz, qw` order;
- `state_7` as the full jaw opening angle in radians.

The constrained jaw skeleton contains a jaw root and two symmetric tips. Each jaw uses `±state_7 / 2`; Tip A opens toward the selected signed opening direction and Tip B opens in the opposite direction. By default the application jointly evaluates all 24 valid orthogonal signed coordinate-axis pairs:

- jaw centerline: `±X`, `±Y`, or `±Z` in the tool frame;
- opening direction: either signed axis orthogonal to the centerline.

Every pair receives an independent robust fit. The pair with the lowest fitting RMS is selected; when Root weight is above `1×`, the corresponding root-weighted fitting RMS is used. Validation landmarks never participate in axis selection. Automatic selection can be disabled in the UI to fix the axes manually, using `+Y` centerline and `+X` opening as the initial values.

The UI also lets the operator choose the:

- jaw-root correction in the tool frame:
  - **No correction** — default;
  - **Manual XYZ correction** — X/Y/Z entered in millimetres;
  - **Optimize one fixed 3D correction** — adds a bounded correction vector shared by every frame.

With zero or manual offset, the only continuously optimized quantities are:

1. one fixed 6-DoF camera extrinsic;
2. one shared jaw length.

Optimize-correction mode additionally fits the three root-correction coordinates. It never fits a separate per-frame 2D correction. The program does not fit a jaw mounting rotation or a gripper gain.

The optional root-landmark weight remains restricted to `1×–3×`. `1×` gives every annotated landmark equal influence; `1.41×` balances the aggregate Root contribution against the two tips; `2×` or `3×` can be used when root alignment is the priority. The weight is applied to fitting frames only and does not change reported, unweighted RMS values. When automatic axis selection is enabled, a weight above `1×` also makes the selector use root-weighted fitting RMS; validation remains excluded.

## Features

- Localhost-only Flask UI; source data are not sent to an external service.
- Session discovery for standard `raw/` and `videos/` layouts or flattened episode folders.
- Strict CSV/video row-count and contiguous-frame checks.
- Left endoscope frame browser, scrubber, magnifier and frame-by-frame playback.
- Manual Root / Tip A / Tip B landmark annotation.
- Uniform frame suggestions or representative frame selection from action data.
- Action-based selection covers low/high gripper values, grasp changes, position, orientation and recording time; action is never used as registration geometry.
- Configurable PSM1/PSM2 observation source.
- Importable OpenCV8 camera intrinsics with optional full-frame resolution scaling.
- Joint discrete optimization over 24 orthogonal signed coordinate-axis pairs, with an optional manual-axis mode.
- Separate fitting and held-out validation frames; validation is excluded from both parameter fitting and axis selection.
- SQPnP camera initialization followed by bounded robust soft-L1 optimization.
- Fit RMS, validation RMS and per-frame residual table.
- Saved landmark JSON includes dataset fingerprint, image size, arm, camera and geometry settings.
- Full registration JSON export with the transform, geometry, residuals and all projected points.
- Full-length MP4 overlay export at source FPS.
- No optical flow, per-frame image snapping, temporal-lag fitting or hidden tracking.

## Install and run

Python 3.10 or newer is required.

### Option A: clone and use a virtual environment

```bash
git clone https://github.com/lala-sean/Sync-demo.git
cd Sync-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
sync-demo
```

Open <http://127.0.0.1:8769>.

The convenience launcher is equivalent:

```bash
sh run.sh
```

### Option B: install directly from GitHub

```bash
python3 -m venv sync-demo-venv
source sync-demo-venv/bin/activate
python -m pip install "git+https://github.com/lala-sean/Sync-demo.git"
sync-demo
```

Set another port if needed:

```bash
REGISTRATION_PORT=8770 sync-demo
```

The server intentionally binds only to `127.0.0.1`.

## Expected data

Standard session layout:

```text
session_folder/
├── raw/chunk-*/episode_000001.csv
└── videos/chunk-*/observation.images.endoscope.left/episode_000001.mp4
```

A flattened directory containing a matching `episode_*.csv` and `episode_*.mp4` is also supported. ZIP files must be extracted first. Parquet-only input is not supported.

The CSV must contain:

- contiguous zero-based `frame_index`;
- `state_0` through `state_15` for two eight-value arm states;
- optionally `action_0` through `action_15` for action-based frame selection.

For each arm, the eight state values are `[x, y, z, qx, qy, qz, qw, jaw_angle]`. The program assumes positions are metres and `jaw_angle` is the total opening angle in radians.

## Camera JSON

The accepted format is:

```json
{
  "lensmodel": "LENSMODEL_OPENCV8",
  "image_size": [1920, 1080],
  "intrinsics": [
    1450.455178496827,
    1429.339083433449,
    866.9314576837802,
    568.0929141757828,
    -0.0071200513187242446,
    -0.02094696502821218,
    0.0017541009755967043,
    0.007817347723294591,
    0.04635027968924447,
    -0.003534950111442823,
    -0.001274634264479936,
    -0.0023892671778790037
  ]
}
```

The order is `fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6`. Resolution scaling assumes only a full-frame resize; it is invalid for cropping, rectification or changed optics.

## Workflow

1. Open a session folder or use the folder picker.
2. Select an episode and PSM observation stream.
3. Confirm camera intrinsics.
4. Leave automatic coordinate-axis selection enabled, or disable it to fix the centerline and opening direction manually. Select the offset mode.
5. Use uniform samples, action-based samples or manually added frames.
6. Mark Root, Tip A and Tip B. Maintain the same physical A/B identity across frames.
7. Use at least six fitting frames, including four complete Root/A/B frames, and at least two held-out validation frames. Eight or more diverse fitting frames are recommended.
8. Run registration and inspect the held-out residuals and overlay.
9. Save landmarks, download the registration JSON or export an overlay MP4.

Validation frames are excluded from SQPnP initialization and nonlinear optimization. They are used only to report held-out error.

## Output transform

`T_camera_PSMbase` maps column vectors from the selected PSM base frame into the camera frame. Translation is in metres. The registration JSON also records:

- camera matrix and distortion values actually used;
- selected centerline, opening axis, all 24 candidate scores, root weight and correction mode;
- resolved tool-frame root correction (`root_correction_tool_m`; the legacy `pivot_offset_tool_m` alias is retained);
- overall, root-only and tip-only RMS values for fitting and validation frames;
- fitted jaw length;
- annotations and their fit/validation roles;
- projected Root/A/B and tool-frame axes for every source frame;
- per-frame, fitting and validation errors;
- source paths, FPS, image size and dataset fingerprint.

CSV row `i` is paired directly with video frame `i`. Matching counts do not prove physical sensor synchronization, and no lag correction is applied.

## Tests

Install the package in editable mode, then run:

```bash
python -m unittest discover -s tests -v
```

The tests cover camera validation, all offset modes, discrete axis selection, axis constraints, symmetric jaw projection, fit/validation and selection isolation, action-based frame selection and localhost request protection.

## MP4 export note

MP4 export requires an OpenCV build with H.264 `avc1` encoding. Some Linux wheels omit this codec. Annotation, fitting and JSON export still work in that case, while MP4 export returns an explicit error.

## Privacy and limitations

- The app has no telemetry, CDN, remote model or external tracking dependency.
- Folder-picker uploads are copied only into this application's local `outputs/` directory.
- The development server is for one local user, not public or multi-user hosting.
- Root annotations, FK error, joint backlash, camera calibration, timing error and inaccurate instrument geometry can all contribute to residual error.
- A held-out split from the same episode is useful for checking overfit but is not an independent calibration validation.

## License

MIT
