# Data preparation

Download the synthetic and real datasets from [EventNeRF](https://github.com/r00tman/EventNeRF). Both use the same scene layout; for example, `nerf/lego` is synthetic and `nextnextgen/plant` is real.

To use your own data, organize each scene as follows:

```text
SCENE/
├── train/
│   ├── events/events.npz
│   ├── pose/00000.txt
│   ├── intrinsics/00000.txt
│   └── rgb/00000.png
└── validation/
    ├── pose/00000.txt
    ├── intrinsics/00000.txt
    └── rgb/00000.png
```

- **Events:** one NPZ file containing equal-length arrays `x`, `y`, `t`, and `p`. Pixel coordinates must match the image resolution, timestamps must be sorted, and polarities may be `{-1,+1}` or `{0,1}`.
- **Poses:** one 4×4 OpenCV camera-to-world matrix per view, with x right, y down, and z forward.
- **Intrinsics:** one 4×4 matrix per view, with the camera calibration matrix K in its upper-left 3×3 block. EventNeRF real-data 5×4 calibration files are also accepted. Image and event coordinates must be rectified consistently; the renderer does not apply distortion coefficients.
- **File matching:** pose, intrinsics, and RGB filenames must have identical stems within each split. Validation RGB images provide the evaluation targets. Training RGB images are optional for event-only reconstruction; if absent, pass `--image_width` and `--image_height`.
- **Timing:** poses are assumed uniformly spaced between zero and the last event timestamp. For nonuniform poses, provide `train/timestamps.txt` with one timestamp per training pose, in filename order and in the same units as the events.

Pass the scene directory to `scripts/train.sh` or `scripts/train_progressive.sh`. Use `--monochrome` for grayscale events; color events use an RGGB Bayer pattern.

## Appearance refinement

For optional refinement, provide blurry RGB images and a JSON manifest listing the camera poses sampled during each exposure:

```json
{
  "fx": 480.55,
  "fy": 480.55,
  "frames": [{
    "image_path": "blur_00000.png",
    "poses": [
      [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    ]
  }]
}
```

Replace the example pose with the calibrated poses covering the exposure, expressed in the event model's world coordinates. Image paths are relative to the manifest. Exposure poses are equally weighted, so sample them uniformly in time.

For EventNeRF RGB sequences, `python scripts/prepare_synthetic_blur.py -s SCENE --output data/blur` creates averaged images and their manifest. Run refinement with `bash scripts/refine.sh EVENT_MODEL data/blur/manifest.json OUTPUT_DIR`.
