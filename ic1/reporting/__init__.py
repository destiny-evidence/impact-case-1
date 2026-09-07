"""Reporting layer: load experiment outputs and render figures.

Design: pure plot functions (df -> matplotlib Figure) shared across domains
(taxonomy, in/out). Heavy compute lives here and runs locally; the mystmd docs
site and PowerPoint both consume the committed figure files, so they can never
disagree. See ic1/reporting/build_figures.py for the render entrypoint.
"""
