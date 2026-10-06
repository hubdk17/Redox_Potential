# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "pymol-open-source-whl",
# ]
# ///

import os
import sys

# Set environment variable for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pymol
pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()

from pymol import cmd

# Load structure
structure_path = "d:/Desktop/JCIM_Paper_FeS/pdb-files/4fxc.pdb"
cmd.load(structure_path, "structure")

# Verify structure loaded
atom_count = cmd.count_atoms("all")
print(f"Loaded structure with {atom_count} atoms.")
if atom_count == 0:
    print("Error: Structure failed to load!")
    cmd.quit()
    sys.exit(1)

# PyMOL styling
cmd.bg_color("white")
cmd.show("cartoon")
cmd.color("gray90", "structure")

# Color secondary structure
cmd.color("red", "ss h") # helices
cmd.color("yellow", "ss s") # sheets

# Highlight cofactor FES (Iron-sulfur cluster)
cmd.show("spheres", "resn FES")
cmd.color("orange", "resn FES and name FE*")
cmd.color("yellow", "resn FES and name S*")

# Highlight coordinating residues (typically CYS)
# Find residues within 4 angstroms of FES
cmd.select("coordinating", "byres (structure and not resn FES) within 4.0 of resn FES")
cmd.show("sticks", "coordinating")
cmd.show("spheres", "coordinating and name SG")
cmd.color("green", "coordinating")

# Zoom and orient
cmd.zoom("resn FES", buffer=8.0)
cmd.orient("resn FES")

# Output rendering
output_png = "C:/Users/HP/.gemini/antigravity-ide/brain/b2fd34b2-acc7-4393-9ee9-9b678224ab55/4fxc_structure.png"
output_pse = "C:/Users/HP/.gemini/antigravity-ide/brain/b2fd34b2-acc7-4393-9ee9-9b678224ab55/4fxc_structure.pse"

cmd.set("ray_opaque_background", 1)
cmd.png(output_png, width=1200, height=900, dpi=150)
cmd.save(output_pse)
cmd.quit()
print("Rendering and session saved successfully.")
