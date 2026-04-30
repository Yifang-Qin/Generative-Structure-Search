import argparse
import os
import re
import subprocess
from glob import glob


def parse_composition(composition):
    pattern = r"([A-Z][a-z]*)(\d*)"
    elements = re.findall(pattern, composition)
    return {el: int(num) for el, num in elements}


def generate_rss_cell_file(composition, varvol):
    composition_dict = parse_composition(composition)
    species_line = "#SPECIES=" + ",".join(f"{el}%NUM={num}" for el, num in composition_dict.items())
    varvol_line = f"#VARVOL={varvol}"
    rss_cell_content = f"""{varvol_line}

{species_line}
#SYMMOPS=1
#NFORM=1
#SLACK=0.25
#OVERLAP=0.1
#MINSEP=1.5
#COMPACT
#CELLADAPT

KPOINTS_MP_SPACING 0.07

SYMMETRY_GENERATE
SNAP_TO_SYMMETRY

%BLOCK SPECIES_POT
QC5
%ENDBLOCK SPECIES_POT

%BLOCK EXTERNAL_PRESSURE
0 0 0
0 0
0
%ENDBLOCK EXTERNAL_PRESSURE
"""
    return rss_cell_content


def write_rss_cell_file(composition, varvol, output_file):
    content = generate_rss_cell_file(composition, varvol)
    with open(output_file, "w") as f:
        f.write(content)
    print(f"Generated {output_file} with VARVOL={varvol} and composition={composition}.")


def run_airss(num_structures):
    command = f"airss.pl -build -max {num_structures} -seed rss"
    print(f"Running AIRSS with command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        print("AIRSS completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running AIRSS: {e}")


def collect_res_files_and_write_xyz(composition, output_file):
    from ase.io import read, write

    res_files = [f for f in os.listdir(".") if f.endswith(".res")]
    if not res_files:
        print("No .res files found in the current directory.")
        return

    atoms_list = []
    for res_file in res_files:
        try:
            atoms = read(res_file)
            atoms_list.append(atoms)
        except Exception as e:
            print(f"Error reading {res_file}: {e}")
    write(output_file, atoms_list)
    print(f"Written {len(atoms_list)} structures to {output_file}.")

    for res_file in res_files:
        os.remove(res_file)
    print(f"Removed {len(res_files)} .res files.")

    # remove all *.cell* file
    for cell_file in glob("*.cell*"):
        os.remove(cell_file)


def main(args):
    write_rss_cell_file(args.composition, args.varvol, "rss.cell")
    run_airss(args.num)

    output_file = args.output or f"{args.composition}_RSS_initial.xyz"
    collect_res_files_and_write_xyz(args.composition, output_file)


if __name__ == "__main__":
    # usage: python gen_rss_comp.py <composition> --num <num_structures> --output <output_file_path>
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "composition", type=str, help="Chemical composition (e.g., Mg1O1, do not omit the number)."
    )
    parser.add_argument(
        "--num", type=int, default=100, help="Number of structures to generate (default: 100)."
    )
    parser.add_argument("--varvol", type=int, default=20, help="VARVOL of volume (default: 20).")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Name of the output .xyz file (default: composition_RSS_initial.xyz)",
    )
    args = parser.parse_args()

    main(args)
