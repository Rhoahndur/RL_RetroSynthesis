"""Prepare a large buyable stock list for retrosynthesis training.

Tries to download from public sources first; falls back to expanding
the existing buyables.csv with curated common building blocks.

Usage:
    python scripts/prepare_stock.py

Outputs:
    data/stock/buyables_full.smi.gz
    environments/retrosynthesis/data/buyables.smi.gz
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure RDKit is available
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
except ImportError:
    print("ERROR: RDKit is required.  pip install rdkit")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stock" / "buyables.csv"
OUT_PRIMARY = ROOT / "data" / "stock" / "buyables_full.smi.gz"
OUT_ENV = ROOT / "environments" / "retrosynthesis" / "data" / "buyables.smi.gz"

# ---------------------------------------------------------------------------
# Public download URLs (tried in order)
# ---------------------------------------------------------------------------
ASKCOS_BUYABLES_URL = "https://github.com/ASKCOS/askcos-data/raw/main/buyables/buyables.json.gz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def canonicalize(smi: str) -> str | None:
    """Return canonical SMILES or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def load_csv(path: Path) -> set[str]:
    """Load SMILES from the existing buyables.csv."""
    smiles: set[str] = set()
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return smiles
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                canon = canonicalize(row[0].strip())
                if canon:
                    smiles.add(canon)
    print(f"  Loaded {len(smiles)} from {path.name}")
    return smiles


def try_download_askcos() -> set[str] | None:
    """Download ASKCOS buyables (~280k compounds) from GitHub."""
    print("Attempting ASKCOS buyables download...")
    try:
        req = urllib.request.Request(ASKCOS_BUYABLES_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
        data = gzip.decompress(raw)
        entries = json.loads(data)
        print(f"  Downloaded {len(entries)} entries from ASKCOS")
        smiles: set[str] = set()
        invalid = 0
        for entry in entries:
            smi = entry.get("smiles", "")
            if not smi:
                continue
            canon = canonicalize(smi)
            if canon:
                smiles.add(canon)
            else:
                invalid += 1
        print(f"  Canonicalized {len(smiles)} valid SMILES ({invalid} invalid/skipped)")
        if len(smiles) > 1000:
            return smiles
        print(f"  Only got {len(smiles)} valid SMILES -- not enough, skipping")
        return None
    except Exception as exc:
        print(f"  Download failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Curated building-block expansions
# ---------------------------------------------------------------------------

# Each category maps to a list of SMILES strings

ALKYL_HALIDES = [
    # methyl
    "CCl",
    "CBr",
    "CI",
    # ethyl
    "CCCl",
    "CCBr",
    "CCI",
    # propyl
    "CCCCl",
    "CCCBr",
    "CCCI",
    # isopropyl
    "CC(C)Cl",
    "CC(C)Br",
    "CC(C)I",
    # butyl
    "CCCCCl",
    "CCCCBr",
    "CCCCI",
    # isobutyl
    "CC(C)CCl",
    "CC(C)CBr",
    "CC(C)CI",
    # sec-butyl
    "CCC(C)Cl",
    "CCC(C)Br",
    "CCC(C)I",
    # tert-butyl
    "CC(C)(C)Cl",
    "CC(C)(C)Br",
    "CC(C)(C)I",
    # benzyl
    "ClCc1ccccc1",
    "BrCc1ccccc1",
    "ICc1ccccc1",
    # allyl
    "C=CCCl",
    "C=CCBr",
    "C=CCI",
    # propargyl
    "C#CCCl",
    "C#CCBr",
    "C#CCI",
    # pentyl / hexyl
    "CCCCCCl",
    "CCCCCBr",
    "CCCCCCCl",
    "CCCCCCBr",
]

ALCOHOLS_AND_PHENOLS = [
    "O",
    "CO",
    "CCO",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "CC(C)O",
    "CC(C)CO",
    "CCC(C)O",
    "CC(C)(C)O",
    "C=CCO",  # allyl alcohol
    "C#CCO",  # propargyl alcohol
    "OCc1ccccc1",  # benzyl alcohol
    "OCC=C",  # allyl alcohol
    "OCCO",  # ethylene glycol
    "OCC(O)CO",  # glycerol
    "Oc1ccccc1",  # phenol
    "Cc1ccc(O)cc1",  # p-cresol
    "Oc1ccc(O)cc1",  # hydroquinone
    "Oc1cccc(O)c1",  # resorcinol
    "Oc1ccccc1O",  # catechol
    "Oc1cc(O)cc(O)c1",  # phloroglucinol
    "Oc1ccc(Cl)cc1",  # 4-chlorophenol
    "Oc1ccc(Br)cc1",  # 4-bromophenol
    "Oc1ccc([N+](=O)[O-])cc1",  # 4-nitrophenol
    "Oc1ccc(F)cc1",  # 4-fluorophenol
    "COc1ccc(O)cc1",  # 4-methoxyphenol
    "Oc1cccc(Cl)c1",  # 3-chlorophenol
    "Oc1ccccc1Cl",  # 2-chlorophenol
    "Oc1c(Cl)cc(Cl)cc1Cl",  # 2,4,6-trichlorophenol
    "Oc1ccncc1",  # 3-hydroxypyridine
    "Oc1ccccn1",  # 2-hydroxypyridine
    "CC(O)c1ccccc1",  # 1-phenylethanol
    "OC(c1ccccc1)c1ccccc1",  # benzhydrol
    "OC(C)(C)C",  # tert-butanol
    "OCCN",  # ethanolamine
    "OCCNCC",  # N-ethylethanolamine
    "OCCO",  # ethylene glycol
    "OC1CCCCC1",  # cyclohexanol
]

AMINO_ACIDS = [
    # 20 natural amino acids (L-form)
    "NCC(=O)O",  # glycine
    "C[C@@H](N)C(=O)O",  # L-alanine
    "CC(C)[C@@H](N)C(=O)O",  # L-valine
    "CC(C)C[C@@H](N)C(=O)O",  # L-leucine
    "CC[C@H](C)[C@@H](N)C(=O)O",  # L-isoleucine
    "OC[C@@H](N)C(=O)O",  # L-serine
    "C[C@H](O)[C@@H](N)C(=O)O",  # L-threonine
    "N[C@@H](Cc1ccccc1)C(=O)O",  # L-phenylalanine
    "N[C@@H](Cc1ccc(O)cc1)C(=O)O",  # L-tyrosine
    "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",  # L-tryptophan
    "N[C@@H](CC(=O)O)C(=O)O",  # L-aspartic acid
    "N[C@@H](CCC(=O)O)C(=O)O",  # L-glutamic acid
    "N[C@@H](CC(N)=O)C(=O)O",  # L-asparagine
    "N[C@@H](CCC(N)=O)C(=O)O",  # L-glutamine
    "N[C@@H](CCCCN)C(=O)O",  # L-lysine
    "N=C(N)NCCC[C@@H](N)C(=O)O",  # L-arginine
    "N[C@@H](Cc1c[nH]cn1)C(=O)O",  # L-histidine
    "CSCC[C@@H](N)C(=O)O",  # L-methionine
    "N[C@@H](CS)C(=O)O",  # L-cysteine
    "O=C(O)[C@@H]1CCCN1",  # L-proline
]

CARBOXYLIC_ACIDS = [
    "CC(=O)O",  # acetic acid
    "CCC(=O)O",  # propionic acid
    "CCCC(=O)O",  # butyric acid
    "CCCCC(=O)O",  # valeric acid
    "CC(C)C(=O)O",  # isobutyric acid
    "OC(=O)c1ccccc1",  # benzoic acid
    "OC(=O)c1ccccc1O",  # salicylic acid
    "OC(=O)c1ccc(O)cc1",  # 4-hydroxybenzoic acid
    "OC(=O)c1ccc(N)cc1",  # 4-aminobenzoic acid
    "OC(=O)c1ccc(Cl)cc1",  # 4-chlorobenzoic acid
    "OC(=O)c1ccc(Br)cc1",  # 4-bromobenzoic acid
    "OC(=O)c1ccc(F)cc1",  # 4-fluorobenzoic acid
    "OC(=O)c1ccc([N+](=O)[O-])cc1",  # 4-nitrobenzoic acid
    "OC(=O)c1ccc(C)cc1",  # 4-methylbenzoic acid
    "OC(=O)c1ccc(OC)cc1",  # 4-methoxybenzoic acid
    "OC(=O)c1cccc(O)c1",  # 3-hydroxybenzoic acid
    "OC(=O)c1cccc(Cl)c1",  # 3-chlorobenzoic acid
    "OC(=O)c1ccccc1Cl",  # 2-chlorobenzoic acid
    "OC(=O)CC(=O)O",  # malonic acid
    "OC(=O)CCC(=O)O",  # succinic acid
    "OC(=O)CCCC(=O)O",  # glutaric acid
    "OC(=O)/C=C/C(=O)O",  # fumaric acid
    "OC(=O)/C=C\\C(=O)O",  # maleic acid
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",  # citric acid
    "OC(=O)C(O)C(O)C(=O)O",  # tartaric acid
    "OC(=O)/C=C/c1ccccc1",  # cinnamic acid
    "OC(=O)CO",  # glycolic acid
    "OC(=O)C(O)C",  # lactic acid
    "OC(=O)CF",  # fluoroacetic acid
    "OC(=O)CCl",  # chloroacetic acid
    "OC(=O)CBr",  # bromoacetic acid
    "OC(=O)C(Cl)Cl",  # dichloroacetic acid
    "OC(=O)C(F)(F)F",  # trifluoroacetic acid
    "C=CC(=O)O",  # acrylic acid
    "CC(=C)C(=O)O",  # methacrylic acid
    "OC(=O)c1ccncc1",  # isonicotinic acid
    "OC(=O)c1cccnc1",  # nicotinic acid
]

AMINES = [
    "N",  # ammonia
    "CN",  # methylamine
    "CCN",  # ethylamine
    "CCCN",  # propylamine
    "CCCCN",  # butylamine
    "CC(C)N",  # isopropylamine
    "CC(C)CN",  # isobutylamine
    "CC(C)(C)N",  # tert-butylamine
    "C1CCNCC1",  # piperidine
    "C1CNCCN1",  # piperazine
    "C1CCNC1",  # pyrrolidine
    "C1COCCN1",  # morpholine
    "CCN(CC)CC",  # triethylamine
    "CN(C)C",  # trimethylamine
    "C1CCNCC1",  # piperidine
    "c1cc[nH]c1",  # pyrrole
    "Nc1ccccc1",  # aniline
    "Nc1ccc(C)cc1",  # p-toluidine
    "Nc1ccc(Cl)cc1",  # 4-chloroaniline
    "Nc1ccc(Br)cc1",  # 4-bromoaniline
    "Nc1ccc(F)cc1",  # 4-fluoroaniline
    "Nc1ccc(OC)cc1",  # p-anisidine
    "Nc1ccc(O)cc1",  # 4-aminophenol
    "Nc1ccc([N+](=O)[O-])cc1",  # 4-nitroaniline
    "Nc1cccc(N)c1",  # m-phenylenediamine
    "Nc1ccc(N)cc1",  # p-phenylenediamine
    "Nc1ccccc1N",  # o-phenylenediamine
    "NCCN",  # ethylenediamine
    "NCCCN",  # 1,3-diaminopropane
    "NCCCCN",  # 1,4-diaminobutane
    "NCCCCCCN",  # 1,6-diaminohexane
    "NCCc1ccccc1",  # phenethylamine
    "NCC(O)=O",  # glycine
    "NCC1CC1",  # cyclopropylmethylamine
    "NC1CCCCC1",  # cyclohexylamine
    "NCc1ccccc1",  # benzylamine
    "CNCc1ccccc1",  # N-methylbenzylamine
    "CNCCNC",  # N,N'-dimethylethylenediamine
    "C(Nc1ccccc1)c1ccccc1",  # N-phenylbenzylamine
    "c1ccc(NCc2ccccc2)cc1",  # N-benzylaniline
]

ALDEHYDES_AND_KETONES = [
    "C=O",  # formaldehyde
    "CC=O",  # acetaldehyde
    "CCC=O",  # propionaldehyde
    "CCCC=O",  # butyraldehyde
    "CC(C)C=O",  # isobutyraldehyde
    "O=Cc1ccccc1",  # benzaldehyde
    "O=Cc1ccc(O)cc1",  # 4-hydroxybenzaldehyde
    "O=Cc1ccc(OC)cc1",  # anisaldehyde
    "O=Cc1ccc(Cl)cc1",  # 4-chlorobenzaldehyde
    "O=Cc1ccc(Br)cc1",  # 4-bromobenzaldehyde
    "O=Cc1ccc(F)cc1",  # 4-fluorobenzaldehyde
    "O=Cc1ccc([N+](=O)[O-])cc1",  # 4-nitrobenzaldehyde
    "O=Cc1ccc(C)cc1",  # 4-methylbenzaldehyde
    "O=Cc1ccccc1O",  # salicylaldehyde
    "O=Cc1ccc(O)c(OC)c1",  # vanillin
    "O=Cc1cccc(OC)c1OC",  # o-veratraldehyde
    "O=Cc1ccncc1",  # isonicotinaldehyde
    "O=Cc1cccnc1",  # nicotinaldehyde
    "O=CC=O",  # glyoxal
    "O=C/C=C/c1ccccc1",  # cinnamaldehyde
    "CC(=O)C",  # acetone
    "CCC(=O)CC",  # 3-pentanone
    "CCC(=O)C",  # methyl ethyl ketone (butanone)
    "CCCC(=O)C",  # 2-pentanone
    "O=C1CCCCC1",  # cyclohexanone
    "O=C1CCCC1",  # cyclopentanone
    "CC(=O)c1ccccc1",  # acetophenone
    "O=C(c1ccccc1)c1ccccc1",  # benzophenone
    "CC(=O)CC(=O)C",  # 2,4-pentanedione (acetylacetone)
    "CC(=O)CC(=O)OCC",  # ethyl acetoacetate
    "CCOC(=O)CC(=O)OCC",  # diethyl malonate
]

BORONIC_ACIDS = [
    "B(O)O",  # boronic acid
    "OB(O)c1ccccc1",  # phenylboronic acid
    "OB(O)c1ccc(C)cc1",  # 4-methylphenylboronic acid
    "OB(O)c1ccc(OC)cc1",  # 4-methoxyphenylboronic acid
    "OB(O)c1ccc(F)cc1",  # 4-fluorophenylboronic acid
    "OB(O)c1ccc(Cl)cc1",  # 4-chlorophenylboronic acid
    "OB(O)c1ccc(Br)cc1",  # 4-bromophenylboronic acid
    "OB(O)c1ccc(C(F)(F)F)cc1",  # 4-trifluoromethylphenylboronic acid
    "OB(O)c1ccc([N+](=O)[O-])cc1",  # 4-nitrophenylboronic acid
    "OB(O)c1ccc(N)cc1",  # 4-aminophenylboronic acid
    "OB(O)c1ccc(O)cc1",  # 4-hydroxyphenylboronic acid
    "OB(O)c1ccccc1F",  # 2-fluorophenylboronic acid
    "OB(O)c1ccccc1Cl",  # 2-chlorophenylboronic acid
    "OB(O)c1ccccc1OC",  # 2-methoxyphenylboronic acid
    "OB(O)c1cccc(F)c1",  # 3-fluorophenylboronic acid
    "OB(O)c1cccc(Cl)c1",  # 3-chlorophenylboronic acid
    "OB(O)c1cccc(OC)c1",  # 3-methoxyphenylboronic acid
    "OB(O)c1cccc(C(F)(F)F)c1",  # 3-trifluoromethylphenylboronic acid
    "OB(O)c1ccncc1",  # pyridin-4-ylboronic acid
    "OB(O)c1cccnc1",  # pyridin-3-ylboronic acid
    "OB(O)c1ccccn1",  # pyridin-2-ylboronic acid
    "OB(O)c1ccsc1",  # thiophen-3-ylboronic acid
    "OB(O)c1cccs1",  # thiophen-2-ylboronic acid
    "OB(O)c1ccc2ccccc2c1",  # 2-naphthylboronic acid
    "OB(O)c1cccc2ccccc12",  # 1-naphthylboronic acid
    "OB(O)c1cc2ccccc2[nH]1",  # indol-2-ylboronic acid
    "OB(O)c1csc(C)n1",  # 2-methyl-4-thiazoleboronic acid
    "CC1(C)OB(c2ccccc2)OC1(C)C",  # phenylboronic acid pinacol ester
    "CC1(C)OB(c2ccc(F)cc2)OC1(C)C",  # 4-fluorophenylboronic acid pinacol ester
    "CC1(C)OB(c2ccc(Cl)cc2)OC1(C)C",  # 4-chlorophenylboronic acid pinacol ester
]

HETEROCYCLES = [
    "c1ccncc1",  # pyridine
    "c1ccnnc1",  # pyridazine
    "c1ccncn1",  # pyrimidine
    "c1cnncn1",  # 1,2,4-triazine
    "c1ccoc1",  # furan
    "c1ccsc1",  # thiophene
    "c1cc[nH]c1",  # pyrrole
    "c1cnc[nH]1",  # imidazole
    "c1cn[nH]c1",  # pyrazole
    "c1ccno1",  # isoxazole
    "c1ccns1",  # thiazole
    "c1c[nH]nn1",  # 1,2,3-triazole
    "c1cnnn1C",  # 1-methyl-1,2,3-triazole
    "c1ccncc1Cl",  # 2-chloropyridine
    "Clc1ccccn1",  # 2-chloropyridine
    "Brc1ccccn1",  # 2-bromopyridine
    "Clc1ccncc1",  # 3-chloropyridine
    "Clc1ccncc1",  # 4-chloropyridine (duplicate, will dedup)
    "Cc1ccncc1",  # 4-methylpyridine
    "Cc1ccccn1",  # 2-methylpyridine
    "Cc1cccnc1",  # 3-methylpyridine
    "Nc1ccccn1",  # 2-aminopyridine
    "Nc1ccncc1",  # 4-aminopyridine
    "Oc1ccccn1",  # 2-hydroxypyridine
    "c1ccc2[nH]ccc2c1",  # indole
    "c1ccc2ncccc2c1",  # quinoline
    "c1ccc2cnccc2c1",  # isoquinoline
    "c1ccc2[nH]c3ccccc3c2c1",  # carbazole
    "c1cnc2ccccc2n1",  # quinazoline
    "c1cnc2ncccc2n1",  # purine scaffold
    "c1ccc2c(c1)oc1ccccc12",  # dibenzofuran
    "c1ccc2c(c1)sc1ccccc12",  # dibenzothiophene
    "C1=CC2=CC=CC=C2N1",  # indoline
    "C1CC2=CC=CC=C2N1",  # indoline (alt)
    "c1csc(-c2ccccn2)n1",  # 2-(thiophen-2-yl)pyridine
    "Cn1ccnc1",  # 1-methylimidazole
    "Cn1ccnc1C",  # 1,2-dimethylimidazole
    "c1ccc(-c2ccccn2)cc1",  # 2-phenylpyridine
    "c1cnc(-c2ccccc2)nc1",  # 2-phenylpyrimidine
    "c1ccc(-c2cccs2)cc1",  # 2-phenylthiophene
    "c1ccc(-c2ccco2)cc1",  # 2-phenylfuran
    # N-heterocyclic fragments (saturated)
    "C1CCNCC1",  # piperidine
    "C1CNCCN1",  # piperazine
    "C1COCCN1",  # morpholine
    "C1CCNC1",  # pyrrolidine
    "C1COCCO1",  # 1,4-dioxane
    "C1CCOC1",  # THF
    "C1CCOCC1",  # THP
]

PROTECTION_REAGENTS = [
    "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",  # Boc2O (di-tert-butyl dicarbonate)
    "O=C(Cl)OCC1c2ccccc2-c2ccccc21",  # Fmoc-Cl
    "O=C(Cl)OCc1ccccc1",  # Cbz-Cl (benzyl chloroformate)
    "C[Si](C)(C)Cl",  # TMS-Cl
    "CC[Si](CC)(CC)Cl",  # TES-Cl
    "CC(C)(C)[Si](C)(C)Cl",  # TBS-Cl
    "C(=O)(Cl)Cl",  # phosgene
    "C(=O)(OCC)Cl",  # ethyl chloroformate
    "CC(=O)OC(C)=O",  # acetic anhydride
    "O=C1OC(=O)c2ccccc21",  # phthalic anhydride
    "O=C1OC(=O)C=C1",  # maleic anhydride
    "O=C1OC(=O)CC1",  # succinic anhydride
    "CC(C)(C)OC(=O)Cl",  # Boc-Cl
]

COUPLING_REAGENTS_AND_MISC = [
    "On1nnc2ccccc21",  # HOBt
    "On1ncc2ccc([N+](=O)[O-])cc21",  # HOAt-analog
    "CCN=C=NCCCN(C)C",  # EDC
    "O=c1[nH]c(=O)c(=O)[nH]1",  # barbituric acid scaffold
    "OP(=O)(O)O",  # phosphoric acid
    "O=P(Cl)(Cl)Cl",  # POCl3
    "O=S(=O)(O)O",  # sulfuric acid
    "O=S(Cl)(=O)c1ccc(C)cc1",  # tosyl chloride
    "CS(=O)(=O)Cl",  # mesyl chloride
    "O=S(=O)(Cl)C(F)(F)F",  # triflyl chloride
    "CS(C)=O",  # DMSO
    "O=C(N)N",  # urea
    "O=[N+]([O-])O",  # nitric acid
    "Cl",  # HCl
    "Br",  # HBr
    "[OH-].[Na+]",  # NaOH
    "[OH-].[K+]",  # KOH
    "[Na+].[Cl-]",  # NaCl
    "[K+].[I-]",  # KI
    "[Na+].[H-]",  # NaH
    "[Li+].CCCC[Li+]",  # n-BuLi (approximate)
    "CCCC[Li]",  # n-BuLi
    "CC(C)(C)[O-].[K+]",  # KOtBu
    "CCOC(=O)OCC",  # diethyl carbonate
    # Grignard-accessible halides
    "BrCCBr",  # 1,2-dibromoethane
    "ClCCCl",  # 1,2-dichloroethane
    "BrCCCBr",  # 1,3-dibromopropane
    "ClCCCCl",  # 1,4-dichlorobutane
]

METAL_CATALYSTS_AND_LIGANDS = [
    "[Pd]",  # palladium (representative)
    "c1ccc(P(c2ccccc2)c2ccccc2)cc1",  # triphenylphosphine
    "CC(C)c1cc(C(C)C)c(P(C(C)(C)C)C(C)(C)C)c(C(C)C)c1",  # XPhos-like
    "c1ccc(-c2ccc3ccccc3c2P(c2ccccc2)c2ccccc2)cc1",  # BINAP-fragment
    "CC(=O)[O-].[Pd+2].CC(=O)[O-]",  # Pd(OAc)2
    "[Cu+].[I-]",  # CuI
    "[Cu+2].[O-]S(=O)(=O)[O-]",  # CuSO4
    "Cl[Sn](Cl)Cl",  # SnCl3 (approx)
    "Cl[Zn]Cl",  # ZnCl2
    "Cl[Ti](Cl)(Cl)Cl",  # TiCl4
    "Cl[Al](Cl)Cl",  # AlCl3
    "Cl[Fe](Cl)Cl",  # FeCl3
    "O=[Ce](=O)=O",  # CeO2 (approximate)
    "[Mg+2].[Cl-].[Cl-]",  # MgCl2
    "[Ca+2].[Cl-].[Cl-]",  # CaCl2
    "[Ag+].[O-][N+](=O)=O",  # AgNO3
]

COMMON_ESTERS_AND_ETHERS = [
    "COC",  # dimethyl ether
    "CCOCC",  # diethyl ether
    "CCCCOCCCC",  # dibutyl ether
    "C1CCOC1",  # THF
    "C1COCCO1",  # 1,4-dioxane
    "COc1ccccc1",  # anisole
    "COC(=O)C",  # methyl acetate
    "CCOC(=O)C",  # ethyl acetate
    "CC(=O)OCC",  # ethyl acetate (alt)
    "CCOC(=O)CC",  # ethyl propanoate
    "COC(=O)c1ccccc1",  # methyl benzoate
    "CCOC(=O)c1ccccc1",  # ethyl benzoate
    "COC(=O)/C=C/c1ccccc1",  # methyl cinnamate
    "COC(=O)OC",  # dimethyl carbonate
]

COMMON_SOLVENTS_AND_BASES = [
    "ClCCl",  # DCM
    "ClC(Cl)Cl",  # chloroform
    "ClC(Cl)(Cl)Cl",  # carbon tetrachloride
    "CC#N",  # acetonitrile
    "CN(C)C=O",  # DMF
    "CN(C)C(=O)C",  # DMAc (N,N-dimethylacetamide)
    "CS(C)=O",  # DMSO
    "C1CCOC1",  # THF
    "CCCCCC",  # hexane
    "c1ccccc1",  # benzene
    "Cc1ccccc1",  # toluene
    "CC(C)c1ccccc1",  # isopropylbenzene (cumene)
    "CCc1ccccc1",  # ethylbenzene
    "c1ccc(Cc2ccccc2)cc1",  # diphenylmethane
    "c1ccc(-c2ccccc2)cc1",  # biphenyl
    "c1ccc2ccccc2c1",  # naphthalene
    "CCN(CC)CC",  # triethylamine
    "c1ccncc1",  # pyridine (base)
    "CN1CCNCC1",  # N-methylpiperazine
    "C1=CN(C)C=N1",  # N-methylimidazole (alt)
    "CC(C)(C)[O-].[K+]",  # KOtBu
    "CC(C)(C)[O-].[Na+]",  # NaOtBu
]

ADDITIONAL_BUILDING_BLOCKS = [
    # Acyl chlorides
    "CC(=O)Cl",  # acetyl chloride
    "CCC(=O)Cl",  # propionyl chloride
    "CCCC(=O)Cl",  # butyryl chloride
    "O=C(Cl)c1ccccc1",  # benzoyl chloride
    "O=C(Cl)c1ccc(Cl)cc1",  # 4-chlorobenzoyl chloride
    "O=C(Cl)/C=C/c1ccccc1",  # cinnamoyl chloride
    # Sulfonyl chlorides
    "O=S(=O)(Cl)c1ccccc1",  # benzenesulfonyl chloride
    "O=S(=O)(Cl)c1ccc(C)cc1",  # tosyl chloride
    "CS(=O)(=O)Cl",  # mesyl chloride
    # Isocyanates and isothiocyanates
    "O=C=Nc1ccccc1",  # phenyl isocyanate
    "S=C=Nc1ccccc1",  # phenyl isothiocyanate
    # Epoxides
    "C1CO1",  # ethylene oxide
    "CC1CO1",  # propylene oxide
    "c1ccc(C2CO2)cc1",  # styrene oxide
    # Alkenes
    "C=C",  # ethylene
    "CC=C",  # propylene
    "C=Cc1ccccc1",  # styrene
    "C=CC=C",  # 1,3-butadiene
    "CC(=C)C",  # isobutylene
    # Alkynes
    "C#C",  # acetylene
    "CC#C",  # propyne
    "C#Cc1ccccc1",  # phenylacetylene
    "C#CCCO",  # propargyl alcohol
    # Nitriles
    "C#N",  # HCN
    "CC#N",  # acetonitrile
    "N#Cc1ccccc1",  # benzonitrile
    "N#CCC#N",  # malononitrile (approx)
    # Misc useful
    "OO",  # hydrogen peroxide
    "O=O",  # O2 (approximate)
    "NN",  # hydrazine
    "NNC(=O)c1ccccc1",  # benzhydrazide
    "ON",  # hydroxylamine
    "CCCCCCCCCCCCCCCCCC(=O)O",  # stearic acid
    "CCCCCCCC/C=C\\CCCCCCCC(=O)O",  # oleic acid
    "OC(=O)CCCCCCCCCCC(=O)O",  # dodecanedioic acid
]


def build_curated_expansion() -> set[str]:
    """Combine all curated building-block categories into a single set of
    canonical SMILES, filtering out anything RDKit rejects."""
    all_raw: list[str] = []
    categories = [
        ("Alkyl halides", ALKYL_HALIDES),
        ("Alcohols/phenols", ALCOHOLS_AND_PHENOLS),
        ("Amino acids", AMINO_ACIDS),
        ("Carboxylic acids", CARBOXYLIC_ACIDS),
        ("Amines", AMINES),
        ("Aldehydes/ketones", ALDEHYDES_AND_KETONES),
        ("Boronic acids", BORONIC_ACIDS),
        ("Heterocycles", HETEROCYCLES),
        ("Protection reagents", PROTECTION_REAGENTS),
        ("Coupling/misc reagents", COUPLING_REAGENTS_AND_MISC),
        ("Metal catalysts/ligands", METAL_CATALYSTS_AND_LIGANDS),
        ("Esters/ethers", COMMON_ESTERS_AND_ETHERS),
        ("Solvents/bases", COMMON_SOLVENTS_AND_BASES),
        ("Additional building blocks", ADDITIONAL_BUILDING_BLOCKS),
    ]

    for name, smiles_list in categories:
        valid = 0
        for smi in smiles_list:
            canon = canonicalize(smi)
            if canon:
                all_raw.append(canon)
                valid += 1
        print(f"  {name}: {valid}/{len(smiles_list)} valid")

    return set(all_raw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Stock List Preparation")
    print("=" * 60)

    all_smiles: set[str] = set()

    # Step 1: Load existing CSV
    print("\n[1/3] Loading existing buyables.csv ...")
    csv_smiles = load_csv(CSV_PATH)
    all_smiles |= csv_smiles

    # Step 2: Try external download
    print("\n[2/3] Trying external downloads ...")
    downloaded = try_download_askcos()
    if downloaded:
        all_smiles |= downloaded
        print(f"  Running total: {len(all_smiles)}")
    else:
        print("  External download unavailable -- using curated expansion only")

    # Step 3: Curated expansion
    print("\n[3/3] Building curated building-block expansion ...")
    curated = build_curated_expansion()
    all_smiles |= curated
    print(f"  Curated expansion added {len(curated)} unique SMILES")

    # Final dedup (already sets, but report)
    print(f"\n  Total unique canonical SMILES: {len(all_smiles)}")

    # Write outputs
    print("\nWriting output files ...")

    for out_path in [OUT_PRIMARY, OUT_ENV]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, "wt") as f:
            for smi in sorted(all_smiles):
                f.write(smi + "\n")
        print(f"  Wrote {out_path}  ({len(all_smiles)} SMILES)")

    print(f"\nDone. {len(all_smiles)} buyable SMILES ready.")


if __name__ == "__main__":
    main()
