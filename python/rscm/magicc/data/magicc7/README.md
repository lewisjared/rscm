# MAGICC7 input files

These are the forcing input files of the MAGICC7 v7.5.3 reference run in the parent directory,
compressed with gzip and otherwise unchanged.
`scripts/regression/generate_full_run.py` copies them from the MAGICC distribution.

- `HISTSSP_*_RF.IN`: historical direct aerosol and black carbon on snow forcing patterns.
- `HISTSSP_*_OT.IN`: historical aerosol optical thickness, used for cloud albedo forcing.
- `GISS_LANDUSE_RF.IN`: historical land-use albedo forcing.
- `SSP245_SOLAR_RF.IN` and `SSP245_VOLCANIC_RF.MON`: RCMIP solar and monthly volcanic forcing.
- `NCYCLE_NFIX3.IN`: terrestrial nitrogen fixation, read when nitrogen limitation is on.

They are part of the MAGICC7 distribution,
Copyright (c) 2019-2021 MAGICC7 Plus IP Co Ltd and MAGICC Contributors (<https://magicc.org>),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
