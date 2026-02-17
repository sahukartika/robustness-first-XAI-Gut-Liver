"""
kegg_pathway_enrichment_shap_hmdb_human.py

Human-Specific KEGG Pathway Enrichment with SHAP-based metabolite selection.
Uses HMDB for synonym lookup to improve KEGG mapping coverage.

Features:
- Reads SHAP values from ALL_SHAP_VALUES.csv
- Multiple threshold options: percentile, percent-based, or absolute SHAP threshold
- Maps metabolites using KEGG FIRST, then falls back to HMDB synonyms
- Filters pathways to HUMAN-SPECIFIC (hsa*) pathways
- Performs pathway enrichment analysis with Fisher's exact test
- Calculates SHAP sum per pathway





"""

import re
import time
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from scipy.stats import fisher_exact


# ==========================
# Helper Functions
# ==========================

def extract_kegg_id_from_feature(feature: str) -> Optional[str]:
    """Extract KEGG compound ID if present in feature name."""
    m = re.search(r'\bC\d{5}\b', feature, re.IGNORECASE)
    if m:
        return m.group(0).upper()
    m = re.search(r'cpd:(C\d{5})', feature, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def strip_lcms_prefix(feature: str) -> str:
    """Remove LC-MS phase prefixes."""
    prefixes = ("fwd_pos_", "fwd_neg_", "rev_pos_", "rev_neg_")
    feature_lower = feature.lower()
    for p in prefixes:
        if feature_lower.startswith(p):
            return feature[len(p):]
    return feature


def normalize_name(name: str) -> str:
    """Normalize metabolite name for comparison."""
    if not name:
        return ""
    normalized = name.lower().strip()
    normalized = ' '.join(normalized.split())
    return normalized


class HMDBClient:
    """
    Client for querying HMDB (Human Metabolome Database).
    Provides synonym lookup and KEGG ID cross-references.
    """

    def __init__(self):
        self.base_url = "https://hmdb.ca"
        self.cache: Dict[str, dict] = {}
        self.name_to_hmdb_cache: Dict[str, Optional[str]] = {}
        
        # Local HMDB database (loaded from file if available)
        self.hmdb_data: Dict[str, dict] = {}  # hmdb_id -> {name, synonyms, kegg_id, ...}
        self.name_to_hmdb: Dict[str, str] = {}  # normalized_name -> hmdb_id
        self._db_loaded = False

    def load_hmdb_database(self, hmdb_file: str = None):
        """
        Load HMDB database from local file for faster lookups.
        """
        if self._db_loaded:
            return True
        
        # Try loading from JSON cache first (faster)
        json_cache = Path("hmdb_metabolites_cache.json")
        if json_cache.exists():
            print("Loading HMDB database from cache...")
            try:
                with open(json_cache, 'r') as f:
                    data = json.load(f)
                    self.hmdb_data = data.get('hmdb_data', {})
                    self.name_to_hmdb = data.get('name_to_hmdb', {})
                self._db_loaded = True
                print(f"Loaded {len(self.hmdb_data)} HMDB entries from cache")
                return True
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Try loading from XML if provided
        if hmdb_file and Path(hmdb_file).exists():
            print(f"Loading HMDB database from {hmdb_file}...")
            print("(This may take several minutes for large files)")
            try:
                self._parse_hmdb_xml(hmdb_file)
                self._db_loaded = True
                
                # Save cache for faster future loads
                self._save_cache(json_cache)
                return True
            except Exception as e:
                print(f"Error loading HMDB XML: {e}")
        
        print("HMDB local database not available. Will use API queries.")
        return False

    def _parse_hmdb_xml(self, xml_file: str):
        """Parse HMDB XML file to extract metabolite info."""
        context = ET.iterparse(xml_file, events=('end',))
        
        count = 0
        for event, elem in context:
            if elem.tag.endswith('metabolite'):
                try:
                    ns = {'hmdb': 'http://www.hmdb.ca'}
                    
                    accession = elem.find('.//accession', ns)
                    if accession is None:
                        accession = elem.find('accession')
                    
                    if accession is not None and accession.text:
                        hmdb_id = accession.text.strip()
                        
                        name_elem = elem.find('.//name', ns)
                        if name_elem is None:
                            name_elem = elem.find('name')
                        primary_name = name_elem.text.strip() if name_elem is not None and name_elem.text else ""
                        
                        synonyms = []
                        syn_container = elem.find('.//synonyms', ns) or elem.find('synonyms')
                        if syn_container is not None:
                            for syn in syn_container.findall('.//synonym', ns) or syn_container.findall('synonym'):
                                if syn.text:
                                    synonyms.append(syn.text.strip())
                        
                        kegg_id = None
                        kegg_elem = elem.find('.//kegg_id', ns) or elem.find('kegg_id')
                        if kegg_elem is not None and kegg_elem.text:
                            kegg_id = kegg_elem.text.strip()
                        
                        iupac_elem = elem.find('.//iupac_name', ns) or elem.find('iupac_name')
                        iupac_name = iupac_elem.text.strip() if iupac_elem is not None and iupac_elem.text else ""
                        
                        self.hmdb_data[hmdb_id] = {
                            'name': primary_name,
                            'iupac_name': iupac_name,
                            'synonyms': synonyms,
                            'kegg_id': kegg_id,
                        }
                        
                        all_names = [primary_name, iupac_name] + synonyms
                        for n in all_names:
                            if n:
                                normalized = normalize_name(n)
                                if normalized and normalized not in self.name_to_hmdb:
                                    self.name_to_hmdb[normalized] = hmdb_id
                        
                        count += 1
                        if count % 10000 == 0:
                            print(f"  Processed {count} metabolites...")
                
                except Exception as e:
                    pass
                
                elem.clear()
        
        print(f"Loaded {len(self.hmdb_data)} metabolites with {len(self.name_to_hmdb)} names")

    def _save_cache(self, cache_file: Path):
        """Save parsed HMDB data to JSON cache."""
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'hmdb_data': self.hmdb_data,
                    'name_to_hmdb': self.name_to_hmdb
                }, f)
            print(f"Saved HMDB cache to {cache_file}")
        except Exception as e:
            print(f"Could not save cache: {e}")

    def search_by_name(self, name: str, timeout: int = 10) -> Optional[dict]:
        """Search HMDB for a metabolite by name."""
        if not name:
            return None
        
        normalized = normalize_name(name)
        
        if self._db_loaded and normalized in self.name_to_hmdb:
            hmdb_id = self.name_to_hmdb[normalized]
            return {
                'hmdb_id': hmdb_id,
                **self.hmdb_data[hmdb_id]
            }
        
        return self._api_search(name, timeout)

    def _api_search(self, name: str, timeout: int = 10) -> Optional[dict]:
        """Search HMDB via API."""
        cache_key = normalize_name(name)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            search_url = f"{self.base_url}/unearth/q?query={name}&searcher=metabolites"
            response = requests.get(search_url, timeout=timeout)
            
            if response.status_code == 200:
                text = response.text
                
                match = re.search(r'href="/metabolites/(HMDB\d+)"', text)
                if match:
                    hmdb_id = match.group(1)
                    
                    details = self._get_metabolite_details(hmdb_id, timeout)
                    if details:
                        self.cache[cache_key] = details
                        return details
        
        except Exception as e:
            pass
        
        self.cache[cache_key] = None
        return None

    def _get_metabolite_details(self, hmdb_id: str, timeout: int = 10) -> Optional[dict]:
        """Get detailed metabolite info from HMDB."""
        try:
            url = f"{self.base_url}/metabolites/{hmdb_id}.xml"
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                result = {'hmdb_id': hmdb_id, 'synonyms': []}
                
                for child in root:
                    tag = child.tag.replace('{http://www.hmdb.ca}', '')
                    
                    if tag == 'name':
                        result['name'] = child.text.strip() if child.text else ""
                    elif tag == 'iupac_name':
                        result['iupac_name'] = child.text.strip() if child.text else ""
                    elif tag == 'kegg_id':
                        result['kegg_id'] = child.text.strip() if child.text else None
                    elif tag == 'synonyms':
                        for syn in child:
                            if syn.text:
                                result['synonyms'].append(syn.text.strip())
                
                return result
        
        except Exception as e:
            pass
        
        return None

    def get_all_names(self, metabolite_info: dict) -> List[str]:
        """Get all possible names for a metabolite."""
        if not metabolite_info:
            return []
        
        names = []
        
        if metabolite_info.get('name'):
            names.append(metabolite_info['name'])
        
        if metabolite_info.get('iupac_name'):
            names.append(metabolite_info['iupac_name'])
        
        names.extend(metabolite_info.get('synonyms', []))
        
        seen = set()
        unique_names = []
        for n in names:
            normalized = normalize_name(n)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_names.append(n)
        
        return unique_names


class KEGGClient:
    """
    KEGG Client with ORGANISM-SPECIFIC pathway filtering.
    
    FIXED:
    - Converts reference pathways (map*) to organism-specific (e.g., hsa*)
    - Uses reference pathways (map*) for compound count queries
    """

    ORGANISM_CODES = {
        'hsa': 'Homo sapiens (human)',
        'mmu': 'Mus musculus (mouse)',
        'rno': 'Rattus norvegicus (rat)',
        'dme': 'Drosophila melanogaster (fruit fly)',
        'cel': 'Caenorhabditis elegans (nematode)',
        'sce': 'Saccharomyces cerevisiae (yeast)',
        'map': 'Reference pathway (all organisms)',
    }

    def __init__(self, organism: str = 'hsa'):
        self.base_url = "https://rest.kegg.jp"
        self.organism = organism.lower()
        
        self.name_to_compound_cache: Dict[str, Optional[str]] = {}
        self.compound_to_pathways_cache: Dict[str, List[str]] = {}
        self.pathway_to_name_cache: Dict[str, Optional[str]] = {}
        self.pathway_compound_counts: Dict[str, int] = {}
        
        self.organism_pathways: Set[str] = set()
        self.organism_compounds: Set[str] = set()
        
        self.total_kegg_compounds = 18000  # Fallback
        
        # KEGG compound database for exact matching
        self.kegg_name_to_id: Dict[str, str] = {}
        self._db_loaded = False
        
        org_name = self.ORGANISM_CODES.get(self.organism, f"Unknown ({self.organism})")
        print(f"\nKEGG Client initialized for: {org_name}")

    def _org_to_ref_pathway(self, pathway_id: str) -> str:
        """
        Convert organism-specific pathway to reference pathway.
        hsa00310 -> map00310
        """
        if pathway_id.startswith(self.organism):
            return "map" + pathway_id[len(self.organism):]
        elif pathway_id.startswith('map'):
            return pathway_id
        else:
            match = re.search(r'(\d{5})$', pathway_id)
            if match:
                return "map" + match.group(1)
            return pathway_id

    def load_organism_pathways(self):
        """Load all valid pathways for the specified organism."""
        print(f"\nLoading {self.organism.upper()} pathway list from KEGG...")
        
        try:
            url = f"{self.base_url}/list/pathway/{self.organism}"
            response = requests.get(url, timeout=60)
            
            if response.status_code != 200:
                print(f"  WARNING: Could not load {self.organism} pathways (HTTP {response.status_code})")
                return
            
            lines = response.text.strip().split('\n')
            
            for line in lines:
                if '\t' not in line:
                    continue
                
                parts = line.split('\t')
                pathway_id = parts[0].replace('path:', '').strip()
                
                self.organism_pathways.add(pathway_id)
                
                if len(parts) > 1:
                    pathway_name = parts[1].strip()
                    if ' - ' in pathway_name:
                        pathway_name = pathway_name.split(' - ')[0].strip()
                    self.pathway_to_name_cache[pathway_id] = pathway_name
            
            print(f"  ✓ Loaded {len(self.organism_pathways)} {self.organism.upper()} pathways")
            
        except Exception as e:
            print(f"  Error loading organism pathways: {e}")

    def load_organism_compounds(self, sleep_between_requests: float = 0.1):
        """
        Load ALL compounds that appear in organism-specific pathways.
        This provides the correct background for enrichment analysis.
        """
        print(f"\nLoading {self.organism.upper()} compound background...")
        print(f"  This may take a few minutes (querying {len(self.organism_pathways)} pathways)...")
        
        self.organism_compounds = set()
        
        for i, pathway_id in enumerate(self.organism_pathways):
            ref_pathway = self._org_to_ref_pathway(pathway_id)
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(self.organism_pathways)} pathways...")
            
            try:
                url = f"{self.base_url}/link/compound/{ref_pathway}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200 and response.text.strip():
                    for line in response.text.strip().split('\n'):
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            compound_id = parts[1].replace('cpd:', '').strip()
                            self.organism_compounds.add(compound_id)
                
                time.sleep(sleep_between_requests)
                
            except Exception as e:
                continue
        
        print(f"  ✓ Found {len(self.organism_compounds)} unique compounds in {self.organism.upper()} pathways")

    def load_kegg_compound_database(self):
        """Load KEGG compound names for exact matching."""
        if self._db_loaded:
            return
        
        print("\nLoading KEGG compound database...")
        
        try:
            url = f"{self.base_url}/list/compound"
            response = requests.get(url, timeout=60)
            
            if response.status_code != 200:
                print("WARNING: Could not load KEGG compound database")
                return
            
            lines = response.text.strip().split('\n')
            print(f"Processing {len(lines)} KEGG compounds...")
            
            for line in lines:
                if '\t' not in line:
                    continue
                
                parts = line.split('\t')
                compound_id = parts[0].replace('cpd:', '').strip()
                
                if len(parts) > 1:
                    names = [n.strip() for n in parts[1].split(';')]
                    for name in names:
                        normalized = normalize_name(name)
                        if normalized and normalized not in self.kegg_name_to_id:
                            self.kegg_name_to_id[normalized] = compound_id
            
            self._db_loaded = True
            self.total_kegg_compounds = len(lines)
            print(f"Loaded {len(self.kegg_name_to_id)} unique compound names")
            
        except Exception as e:
            print(f"Error loading KEGG database: {e}")

    def search_compound_exact(self, name: str) -> Optional[str]:
        """Search for compound by exact name match (case-insensitive)."""
        if not name:
            return None
        
        normalized = normalize_name(name)
        
        if normalized in self.name_to_compound_cache:
            return self.name_to_compound_cache[normalized]
        
        if self._db_loaded and normalized in self.kegg_name_to_id:
            compound_id = self.kegg_name_to_id[normalized]
            self.name_to_compound_cache[normalized] = compound_id
            return compound_id
        
        return None

    def search_compound_api(self, name: str, exact: bool = True, timeout: int = 10) -> Optional[str]:
        """Search KEGG API for compound."""
        if not name:
            return None
        
        normalized = normalize_name(name)
        cache_key = f"api_{normalized}"
        
        if cache_key in self.name_to_compound_cache:
            return self.name_to_compound_cache[cache_key]
        
        query = name.replace(" ", "+")
        url = f"{self.base_url}/find/compound/{query}"
        
        try:
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200 and response.text.strip():
                for line in response.text.strip().split('\n'):
                    if '\t' not in line:
                        continue
                    
                    parts = line.split('\t')
                    entry = parts[0]
                    
                    if not entry.startswith("cpd:"):
                        continue
                    
                    compound_id = entry.split(":", 1)[1].strip()
                    
                    if exact and len(parts) > 1:
                        names = [n.strip() for n in parts[1].split(';')]
                        for kegg_name in names:
                            if normalize_name(kegg_name) == normalized:
                                self.name_to_compound_cache[cache_key] = compound_id
                                return compound_id
                    else:
                        self.name_to_compound_cache[cache_key] = compound_id
                        return compound_id
        
        except Exception:
            pass
        
        self.name_to_compound_cache[cache_key] = None
        return None

    def get_compound_pathways(self, compound_id: str, timeout: int = 10) -> List[str]:
        """
        Get all ORGANISM-SPECIFIC pathways for a compound.
        
        KEGG returns reference pathways (map*), which we convert to 
        organism-specific pathways (e.g., hsa* for human).
        """
        if not compound_id:
            return []
        
        cache_key = f"{compound_id}_{self.organism}"
        if cache_key in self.compound_to_pathways_cache:
            return self.compound_to_pathways_cache[cache_key]
        
        try:
            url = f"{self.base_url}/link/pathway/cpd:{compound_id}"
            response = requests.get(url, timeout=timeout)
            
            pathways = []
            if response.status_code == 200 and response.text.strip():
                for line in response.text.strip().split('\n'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pathway_id = parts[1].replace('path:', '').strip()
                        
                        # Convert reference pathways (map*) to organism-specific
                        if pathway_id.startswith('map'):
                            pathway_num = pathway_id[3:]
                            org_pathway_id = f"{self.organism}{pathway_num}"
                            
                            # Verify this pathway exists for the organism
                            if len(self.organism_pathways) == 0 or org_pathway_id in self.organism_pathways:
                                pathways.append(org_pathway_id)
                        
                        elif pathway_id.startswith(self.organism):
                            pathways.append(pathway_id)
            
            self.compound_to_pathways_cache[cache_key] = pathways
            return pathways
            
        except Exception as e:
            self.compound_to_pathways_cache[cache_key] = []
            return []

    def get_pathway_name(self, pathway_id: str, timeout: int = 10) -> str:
        """Get pathway name."""
        if not pathway_id:
            return pathway_id
        
        if pathway_id in self.pathway_to_name_cache:
            return self.pathway_to_name_cache[pathway_id] or pathway_id
        
        try:
            url = f"{self.base_url}/get/{pathway_id}"
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                for line in response.text.split('\n'):
                    if line.startswith('NAME'):
                        name = line.replace('NAME', '').strip()
                        if ' - ' in name:
                            name = name.split(' - ')[0].strip()
                        self.pathway_to_name_cache[pathway_id] = name
                        return name
        except:
            pass
        
        self.pathway_to_name_cache[pathway_id] = None
        return pathway_id

    def get_pathway_compound_count(self, pathway_id: str, timeout: int = 10) -> int:
        """
        Get number of compounds in a pathway.
        
        FIXED: KEGG only stores compound links for REFERENCE pathways (map*),
        so we must convert organism-specific pathways back to reference format.
        """
        if pathway_id in self.pathway_compound_counts:
            return self.pathway_compound_counts[pathway_id]
        
        # Convert organism pathway to reference pathway
        # hsa00310 -> map00310
        ref_pathway_id = self._org_to_ref_pathway(pathway_id)
        
        try:
            url = f"{self.base_url}/link/compound/{ref_pathway_id}"
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200 and response.text.strip():
                count = len(response.text.strip().split('\n'))
                self.pathway_compound_counts[pathway_id] = count
                return count
        except:
            pass
        
        self.pathway_compound_counts[pathway_id] = 0
        return 0

    def get_total_compounds(self, use_organism_specific: bool = True) -> int:
        """
        Get total compound count for background.
        
        Args:
            use_organism_specific: If True, return compounds in organism pathways.
                                   If False, return all KEGG compounds.
        """
        if use_organism_specific and len(self.organism_compounds) > 0:
            return len(self.organism_compounds)
        return self.total_kegg_compounds

    def get_organism_info(self) -> str:
        return self.ORGANISM_CODES.get(self.organism, f"Custom ({self.organism})")


class KEGGPathwayEnrichmentSHAP:
    """
    KEGG Pathway Enrichment with SHAP-based selection, HMDB synonym support,
    and ORGANISM-SPECIFIC pathway filtering.
    
    SEARCH ORDER:
    1. First try KEGG directly
    2. If not found in KEGG, fall back to HMDB for synonyms
    """

    def __init__(self, organism: str = 'hsa', use_organism_background: bool = True):
        self.organism = organism.lower()
        self.use_organism_background = use_organism_background
        self.hmdb = HMDBClient()
        self.kegg = KEGGClient(organism=self.organism)

    def load_shap_metabolites(
        self,
        shap_file: str,
        shap_column: str = "mean_abs_shap",
        threshold_method: str = "percentile",
        threshold_value: float = 75.0,
    ) -> pd.DataFrame:
        """Load and filter metabolites by SHAP values."""
        shap_path = Path(shap_file)
        if not shap_path.exists():
            raise FileNotFoundError(f"SHAP file not found: {shap_file}")
        
        print(f"\nLoading SHAP file: {shap_file}")
        df = pd.read_csv(shap_file)
        
        if "feature" not in df.columns or shap_column not in df.columns:
            raise ValueError(f"SHAP file must contain 'feature' and '{shap_column}' columns.")
        
        # Aggregate SHAP values per feature
        print("Aggregating SHAP values per feature...")
        agg = (
            df.groupby("feature")[shap_column]
            .mean()
            .reset_index()
            .rename(columns={shap_column: "mean_abs_shap"})
        )
        
        agg = agg.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        
        total_features = len(agg)
        print(f"Total features: {total_features}")
        
        # Filter to non-zero for percent/percentile
        nonzero = agg[agg["mean_abs_shap"] > 0].copy()
        nonzero = nonzero.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        
        n_nonzero = len(nonzero)
        print(f"Features with SHAP > 0: {n_nonzero}")
        print(f"Features with SHAP = 0: {total_features - n_nonzero}")
        
        if threshold_method == "percentile":
            if n_nonzero == 0:
                return pd.DataFrame(columns=["feature", "mean_abs_shap"])
            
            threshold = np.percentile(nonzero["mean_abs_shap"], threshold_value)
            selected = nonzero[nonzero["mean_abs_shap"] >= threshold].copy()
            print(f"\nThreshold: PERCENTILE >= {threshold_value}th of non-zero")
            print(f"SHAP threshold: {threshold:.6f}")
            
        elif threshold_method == "percent":
            if n_nonzero == 0:
                return pd.DataFrame(columns=["feature", "mean_abs_shap"])
            
            n_keep = max(1, int(n_nonzero * threshold_value / 100))
            selected = nonzero.head(n_keep).copy()
            print(f"\nThreshold: TOP {threshold_value}% of non-zero ({n_keep} features)")
            
        elif threshold_method == "absolute":
            selected = agg[agg["mean_abs_shap"] > threshold_value].copy()
            print(f"\nThreshold: ABSOLUTE > {threshold_value}")
            
        else:
            raise ValueError(f"Unknown threshold_method: {threshold_method}")
        
        selected = selected.reset_index(drop=True)
        print(f"Selected features: {len(selected)}")
        
        return selected

    def map_metabolite_to_kegg(
        self,
        feature: str,
        use_hmdb: bool = True,
        sleep_time: float = 0.2,
    ) -> Tuple[Optional[str], dict]:
        """
        Map a metabolite feature to KEGG ID.
        
        SEARCH ORDER:
        1. Check if KEGG ID is in feature name
        2. Try KEGG exact match (local database)
        3. Try KEGG API search
        4. If not found and use_hmdb=True, try HMDB for synonyms
        """
        mapping_info = {
            'original_feature': feature,
            'query_name': strip_lcms_prefix(feature).strip(),
            'hmdb_id': None,
            'hmdb_name': None,
            'synonyms_tried': [],
            'kegg_id': None,
            'kegg_source': None,
        }
        
        query_name = mapping_info['query_name']
        
        # STEP 1: Check if KEGG ID is in feature name
        kegg_id = extract_kegg_id_from_feature(feature)
        if kegg_id:
            mapping_info['kegg_id'] = kegg_id
            mapping_info['kegg_source'] = 'feature_name'
            return kegg_id, mapping_info
        
        # STEP 2: Try KEGG exact match (from local database)
        kegg_id = self.kegg.search_compound_exact(query_name)
        if kegg_id:
            mapping_info['kegg_id'] = kegg_id
            mapping_info['kegg_source'] = 'kegg_exact'
            return kegg_id, mapping_info
        
        # STEP 3: Try KEGG API search
        kegg_id = self.kegg.search_compound_api(query_name, exact=True)
        if kegg_id:
            mapping_info['kegg_id'] = kegg_id
            mapping_info['kegg_source'] = 'kegg_api'
            time.sleep(sleep_time)
            return kegg_id, mapping_info
        
        time.sleep(sleep_time)
        
        # STEP 4: Fall back to HMDB (if enabled)
        if use_hmdb:
            hmdb_info = self.hmdb.search_by_name(query_name)
            
            if hmdb_info:
                mapping_info['hmdb_id'] = hmdb_info.get('hmdb_id')
                mapping_info['hmdb_name'] = hmdb_info.get('name')
                
                # 4a. Check if HMDB has KEGG cross-reference
                kegg_id = hmdb_info.get('kegg_id')
                if kegg_id and kegg_id.startswith('C'):
                    mapping_info['kegg_id'] = kegg_id
                    mapping_info['kegg_source'] = 'hmdb_crossref'
                    return kegg_id, mapping_info
                
                # 4b. Try all HMDB names/synonyms against KEGG
                all_names = self.hmdb.get_all_names(hmdb_info)
                
                for name in all_names:
                    mapping_info['synonyms_tried'].append(name)
                    
                    kegg_id = self.kegg.search_compound_exact(name)
                    if kegg_id:
                        mapping_info['kegg_id'] = kegg_id
                        mapping_info['kegg_source'] = f'hmdb_synonym_exact: {name}'
                        return kegg_id, mapping_info
                
                # 4c. Try KEGG API search with synonyms (limit API calls)
                for name in all_names:
                    kegg_id = self.kegg.search_compound_api(name, exact=True)
                    if kegg_id:
                        mapping_info['kegg_id'] = kegg_id
                        mapping_info['kegg_source'] = f'hmdb_synonym_api: {name}'
                        time.sleep(sleep_time)
                        return kegg_id, mapping_info
                    time.sleep(sleep_time)
        
        return None, mapping_info

    def analyze(
        self,
        shap_file: str,
        output_dir: str = None,
        shap_column: str = "mean_abs_shap",
        threshold_method: str = "percentile",
        threshold_value: float = 75.0,
        use_hmdb: bool = True,
        hmdb_file: str = None,
        sleep_between_requests: float = 0.3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete KEGG pathway enrichment analysis with HUMAN-SPECIFIC pathways.
        """
        # Setup output directory
        if output_dir is None:
            shap_path = Path(shap_file)
            output_dir = shap_path.parent / f"kegg_enrichment_{self.organism}_results"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
        
        # Load databases
        print("\n" + "="*70)
        print("LOADING DATABASES")
        print("="*70)
        
        self.kegg.load_kegg_compound_database()
        
        # Load organism-specific pathways
        self.kegg.load_organism_pathways()
        
        # Optionally load organism-specific compound background
        if self.use_organism_background:
            self.kegg.load_organism_compounds(sleep_between_requests=0.1)
        
        if use_hmdb:
            self.hmdb.load_hmdb_database(hmdb_file)
        
        print(f"\nOrganism: {self.kegg.get_organism_info()}")
        print(f"Background mode: {'Organism-specific' if self.use_organism_background else 'All KEGG'}")
        
        # Step 1: Load metabolites
        print("\n" + "="*70)
        print("STEP 1: Loading and filtering metabolites by SHAP values")
        print("="*70)
        
        selected_metabolites = self.load_shap_metabolites(
            shap_file=shap_file,
            shap_column=shap_column,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
        )
        
        if len(selected_metabolites) == 0:
            print("ERROR: No metabolites selected!")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Get background compound count
        total_background_compounds = self.kegg.get_total_compounds(
            use_organism_specific=self.use_organism_background
        )
        print(f"\nBackground compounds: {total_background_compounds:,}")
        
        # Step 2: Map metabolites to KEGG IDs
        print("\n" + "="*70)
        print("STEP 2: Mapping metabolites to KEGG IDs")
        print("        (KEGG first, HMDB fallback for synonyms)")
        print("="*70)
        
        mapping_records = []
        metabolite_shap = {}  # kegg_id -> list of (feature, shap)
        
        source_counts = defaultdict(int)
        
        for i, row in selected_metabolites.iterrows():
            feature = row["feature"]
            mean_shap = float(row["mean_abs_shap"])
            
            display_name = feature[:45] + "..." if len(feature) > 45 else feature
            print(f"  [{i+1}/{len(selected_metabolites)}] {display_name}", end="")
            
            kegg_id, mapping_info = self.map_metabolite_to_kegg(
                feature=feature,
                use_hmdb=use_hmdb,
                sleep_time=sleep_between_requests,
            )
            
            mapping_info['mean_abs_shap'] = mean_shap
            mapping_info['status'] = 'mapped' if kegg_id else 'not_found'
            mapping_info['synonyms_tried'] = '; '.join(mapping_info['synonyms_tried'][:10])
            
            if kegg_id:
                source = mapping_info['kegg_source']
                if source.startswith('hmdb_synonym'):
                    source_key = 'hmdb_synonym'
                else:
                    source_key = source
                source_counts[source_key] += 1
                
                print(f" -> {kegg_id} ({mapping_info['kegg_source']}) ✓")
                
                if kegg_id not in metabolite_shap:
                    metabolite_shap[kegg_id] = []
                metabolite_shap[kegg_id].append((feature, mean_shap))
            else:
                print(" -> Not found ✗")
            
            mapping_records.append(mapping_info)
        
        kegg_ids = list(metabolite_shap.keys())
        n_mapped = len(kegg_ids)
        n_total = len(selected_metabolites)
        
        print(f"\nMapping Summary:")
        print(f"  Total metabolites: {n_total}")
        print(f"  Mapped to KEGG: {n_mapped} ({100*n_mapped/n_total:.1f}%)")
        print(f"  Not found: {n_total - n_mapped}")
        
        print(f"\nMapping Sources:")
        for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"  {source}: {count}")
        
        # Save mapping file
        mapping_df = pd.DataFrame(mapping_records)
        mapping_file = output_dir / "metabolite_kegg_mapping.csv"
        mapping_df.to_csv(mapping_file, index=False)
        print(f"\nMapping saved to: {mapping_file}")
        
        if not kegg_ids:
            print("ERROR: No metabolites could be mapped to KEGG!")
            return pd.DataFrame(), pd.DataFrame(), mapping_df
        
        # Step 3: Get organism-specific pathways
        print("\n" + "="*70)
        print(f"STEP 3: Finding {self.organism.upper()} pathways for each metabolite")
        print("="*70)
        
        pathway_hits = defaultdict(list)
        pathway_shap = defaultdict(float)
        pathway_metabolites = defaultdict(list)
        
        long_mapping_records = []
        
        for i, kegg_id in enumerate(kegg_ids):
            features_shaps = metabolite_shap[kegg_id]
            total_shap = sum(s for _, s in features_shaps)
            feature_names = [f for f, _ in features_shaps]
            
            print(f"  [{i+1}/{len(kegg_ids)}] {kegg_id}", end=" -> ")
            
            pathways = self.kegg.get_compound_pathways(kegg_id)
            print(f"{len(pathways)} {self.organism} pathways")
            
            if not pathways:
                for feature, shap_val in features_shaps:
                    long_mapping_records.append({
                        'original_feature': feature,
                        'kegg_compound_id': kegg_id,
                        'pathway_id': None,
                        'pathway_name': None,
                        'organism': self.organism,
                        'mean_abs_shap': shap_val,
                    })
            else:
                for pw_id in pathways:
                    pathway_hits[pw_id].append(kegg_id)
                    pathway_shap[pw_id] += total_shap
                    pathway_metabolites[pw_id].extend(feature_names)
                    
                    pw_name = self.kegg.get_pathway_name(pw_id)
                    
                    for feature, shap_val in features_shaps:
                        long_mapping_records.append({
                            'original_feature': feature,
                            'kegg_compound_id': kegg_id,
                            'pathway_id': pw_id,
                            'pathway_name': pw_name,
                            'organism': self.organism,
                            'mean_abs_shap': shap_val,
                        })
                    
                    time.sleep(sleep_between_requests * 0.3)
            
            time.sleep(sleep_between_requests)
        
        print(f"\nFound metabolites in {len(pathway_hits)} unique {self.organism.upper()} pathways")
        
        # Save long mapping
        long_mapping_df = pd.DataFrame(long_mapping_records)
        long_mapping_file = output_dir / f"metabolite_{self.organism}_pathway_long_mapping.csv"
        long_mapping_df.to_csv(long_mapping_file, index=False)
        
        if len(pathway_hits) == 0:
            print(f"\nWARNING: No {self.organism.upper()} pathways found!")
            return pd.DataFrame(), pd.DataFrame(), mapping_df
        
        # Step 4: Enrichment analysis
        print("\n" + "="*70)
        print(f"STEP 4: Calculating {self.organism.upper()} pathway enrichment")
        print(f"        Background: {total_background_compounds:,} compounds")
        print("="*70)
        
        query_size = len(kegg_ids)
        results = []
        
        for i, (pathway_id, hits) in enumerate(pathway_hits.items()):
            print(f"  [{i+1}/{len(pathway_hits)}] {pathway_id}", end=" -> ")
            
            pathway_name = self.kegg.get_pathway_name(pathway_id)
            pathway_size = self.kegg.get_pathway_compound_count(pathway_id)
            
            print(f"Size: {pathway_size}, Name: {pathway_name[:30]}...")
            
            if pathway_size == 0:
                print(f"    WARNING: Skipping {pathway_id} (size=0)")
                continue
            
            hits_unique = list(set(hits))
            hits_count = len(hits_unique)
            hit_features = list(set(pathway_metabolites[pathway_id]))
            
            # Fisher's exact test with correct background
            a = hits_count
            b = query_size - hits_count
            c = pathway_size - hits_count
            d = total_background_compounds - pathway_size - query_size + hits_count
            
            a, b, c, d = max(0, a), max(0, b), max(0, c), max(0, d)
            
            try:
                odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative='greater')
            except:
                odds_ratio, pvalue = 1.0, 1.0
            
            expected = (query_size * pathway_size) / total_background_compounds
            enrichment_ratio = hits_count / expected if expected > 0 else 0
            
            results.append({
                'pathway_id': pathway_id,
                'pathway_name': pathway_name,
                'organism': self.organism,
                'pathway_size': pathway_size,
                'hits_count': hits_count,
                'expected_hits': round(expected, 4),
                'enrichment_ratio': round(enrichment_ratio, 4),
                'shap_sum': round(pathway_shap[pathway_id], 6),
                'hit_metabolites': ', '.join(sorted(hit_features)),
                'hit_kegg_ids': ', '.join(sorted(hits_unique)),
                'pvalue': pvalue,
                'odds_ratio': odds_ratio,
                'background_compounds': total_background_compounds,
            })
            
            time.sleep(sleep_between_requests)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if results_df.empty:
            print("No enrichment results")
            return pd.DataFrame(), pd.DataFrame(), mapping_df
        
        # FDR correction
        results_df = results_df.sort_values('pvalue')
        n = len(results_df)
        results_df['rank'] = range(1, n + 1)
        results_df['fdr'] = (results_df['pvalue'] * n / results_df['rank']).clip(upper=1.0)
        results_df['fdr'] = results_df['fdr'][::-1].cummin()[::-1]
        results_df = results_df.drop('rank', axis=1)
        
        # Reorder columns
        column_order = [
            'pathway_id', 'pathway_name', 'organism', 'pathway_size', 'hits_count',
            'expected_hits', 'enrichment_ratio', 'shap_sum', 'pvalue', 'fdr',
            'odds_ratio', 'background_compounds', 'hit_metabolites', 'hit_kegg_ids'
        ]
        results_df = results_df[[c for c in column_order if c in results_df.columns]]
        
        # Display results
        print("\n" + "="*70)
        print(f"RESULTS: {self.organism.upper()} KEGG PATHWAY ENRICHMENT")
        print("="*70)
        
        significant = results_df[results_df['fdr'] < 0.05].copy()
        
        print(f"\nBackground: {total_background_compounds:,} compounds")
        print(f"Total {self.organism.upper()} pathways: {len(results_df)}")
        print(f"Significant (FDR < 0.05): {len(significant)}")
        
        print("\nTop 20 Pathways:")
        print("-"*120)
        
        for _, row in results_df.head(20).iterrows():
            sig = "***" if row['fdr'] < 0.05 else ""
            name = str(row['pathway_name'])[:35]
            print(f"  {name:<35} Hits: {row['hits_count']:>3}/{row['pathway_size']:<3} "
                  f"Enrich: {row['enrichment_ratio']:>6.2f} SHAP: {row['shap_sum']:.4f} FDR: {row['fdr']:.2e} {sig}")
        
        # Save results
        full_results_file = output_dir / f"{self.organism}_pathway_enrichment_full_results.csv"
        results_df.to_csv(full_results_file, index=False)
        print(f"\nFull results: {full_results_file}")
        
        if len(significant) > 0:
            sig_file = output_dir / f"{self.organism}_pathway_enrichment_significant_FDR05.csv"
            significant.to_csv(sig_file, index=False)
            print(f"Significant results: {sig_file}")
        
        # Save summary
        bg_mode = 'Organism-specific' if self.use_organism_background else 'All KEGG'
        summary_file = output_dir / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"{self.organism.upper()} KEGG Pathway Enrichment Analysis Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Organism: {self.kegg.get_organism_info()}\n")
            f.write(f"Input file: {shap_file}\n")
            f.write(f"HMDB fallback used: {use_hmdb}\n")
            f.write(f"Threshold method: {threshold_method}\n")
            f.write(f"Threshold value: {threshold_value}\n\n")
            f.write(f"Background mode: {bg_mode}\n")
            f.write(f"Background compounds: {total_background_compounds}\n\n")
            f.write(f"Total metabolites: {n_total}\n")
            f.write(f"Mapped to KEGG: {n_mapped} ({100*n_mapped/n_total:.1f}%)\n")
            f.write(f"\nMapping Sources:\n")
            for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {source}: {count}\n")
            f.write(f"\n{self.organism.upper()} Pathways analyzed: {len(results_df)}\n")
            f.write(f"Significant (FDR<0.05): {len(significant)}\n")
        
        # Plot
        self._plot_results(results_df, output_dir)
        
        return results_df, significant, mapping_df

    def _plot_results(self, results_df, output_dir, top_n=20):
        """Create visualization."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        top = results_df.head(top_n).copy()
        if len(top) == 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        
        # Plot 1: -log10(FDR)
        ax1 = axes[0]
        y_vals = -np.log10(top['fdr'].values + 1e-300)
        colors = ['red' if fdr < 0.05 else 'steelblue' for fdr in top['fdr']]
        
        ax1.barh(range(len(top)), y_vals, color=colors)
        ax1.set_yticks(range(len(top)))
        ax1.set_yticklabels([str(n)[:40] for n in top['pathway_name'].values], fontsize=8)
        ax1.set_xlabel('-log10(FDR)')
        ax1.set_title(f'{self.organism.upper()} Pathway Significance')
        ax1.invert_yaxis()
        ax1.axvline(x=-np.log10(0.05), color='green', linestyle='--', label='FDR=0.05')
        ax1.legend()
        
        # Plot 2: Enrichment
        ax2 = axes[1]
        ax2.barh(range(len(top)), top['enrichment_ratio'].values, color=colors)
        ax2.set_yticks(range(len(top)))
        ax2.set_yticklabels([str(n)[:40] for n in top['pathway_name'].values], fontsize=8)
        ax2.set_xlabel('Enrichment Ratio')
        ax2.set_title('Enrichment Ratio')
        ax2.invert_yaxis()
        ax2.axvline(x=1.0, color='gray', linestyle='--')
        
        # Plot 3: SHAP
        ax3 = axes[2]
        ax3.barh(range(len(top)), top['shap_sum'].values, color=colors)
        ax3.set_yticks(range(len(top)))
        ax3.set_yticklabels([str(n)[:40] for n in top['pathway_name'].values], fontsize=8)
        ax3.set_xlabel('SHAP Sum')
        ax3.set_title('SHAP Importance')
        ax3.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.organism}_pathway_enrichment_plot.png', dpi=150, bbox_inches='tight')
        plt.close()


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    # ======================================================================
    # CONFIGURATION
    # ======================================================================
    
    SHAP_FILE = "ALL_SHAP_VALUES.csv"
    OUTPUT_DIR = "pathways"
    
    SHAP_COLUMN = "mean_abs_shap"
    THRESHOLD_METHOD = "absolute"  # "percentile", "percent", "absolute"
    THRESHOLD_VALUE = 0  # All non-zero SHAP values
    
    # ORGANISM SETTINGS
    ORGANISM = 'hsa'  # Human (Homo sapiens)
    USE_ORGANISM_BACKGROUND = True  # Use human-specific compound background
    
    # HMDB settings - used as FALLBACK (KEGG searched first)
    USE_HMDB = True
    HMDB_FILE = None  # Option is there without being used in this study
    
    SLEEP_BETWEEN_REQUESTS = 1
    
    # ======================================================================
    # RUN
    # ======================================================================
    
    print("\n" + "="*70)
    print("   HUMAN-SPECIFIC KEGG PATHWAY ENRICHMENT")
    print("   WITH SHAP-BASED METABOLITE SELECTION")
    print("="*70)
    
    analyzer = KEGGPathwayEnrichmentSHAP(
        organism=ORGANISM,
        use_organism_background=USE_ORGANISM_BACKGROUND,
    )
    
    full_results, significant, mapping = analyzer.analyze(
        shap_file=SHAP_FILE,
        output_dir=OUTPUT_DIR,
        shap_column=SHAP_COLUMN,
        threshold_method=THRESHOLD_METHOD,
        threshold_value=THRESHOLD_VALUE,
        use_hmdb=USE_HMDB,
        hmdb_file=HMDB_FILE,
        sleep_between_requests=SLEEP_BETWEEN_REQUESTS,
    )
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)