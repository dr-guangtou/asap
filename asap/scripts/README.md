# Prepare the UniverseMachine model for running ASAP model 

----

## Requirements: 

1. ASCII output file from `UniverseMachine` model (e.g., `sfr_catalog_0.712400.txt`)
2. Optional: ASCII halo catalog from `RockStar` model (e.g., `hlist_0.71240.list`)
    - If additional halo properties are necessary, please make sure the scale factors match for the two files.

## Basic steps:

1. `um_organize_snapshot.py`: Reduce the file size by keep a few useful columns and only include halo above
   certain `M_peak` cut; also convert the catalog into `numpy` file. Here the centrals and satellites are kept
   in different records: `centrals` & `satellites`.
2. `um_merge_with_hlist.py`: [Optional] Merge with the original `Rockstar` halo catalog to include additional
   halo information.
3. `um_make_galaxy_catalog.py`: [Unfinished] Make galaxy and halo catalog for `ASAP` model.
4. `um_smdpl_downsample_particles.py`: [Unfinished] Downsample the partile list from the dark matter simulation.
5. `um_smdpl_precompute_dsigma.py`: [Unfinished] Precomput the mass enclosed in different radius around a galaxy.
