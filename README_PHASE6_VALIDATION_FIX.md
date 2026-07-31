# Phase 6 validation fix

This patch fixes two validation issues:

1. Structured-report defect labels are compared semantically. A source metadata
   value such as `bubble,discolor,scratch` is considered equivalent to the
   human-readable JSON value `bubble, discolor, scratch`.
2. The sample gallery now selects `phase6_gallery_samples_per_family` records
   independently for every supported task family, ensuring all eight task
   families appear.

Extract the patch into the project root, overwrite matching files, run tests,
then rerun Phase 6.
