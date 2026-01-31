# loader.py
from statsbombpy import sb
import warnings

# This silences the NoAuthWarning if you find it annoying
warnings.filterwarnings("ignore", message="credentials were not supplied")

class StatsBombLoader:
    """
    Responsible for fetching metadata about competitions and matches,
    specifically filtering for those with StatsBomb 360 data.
    """
    
    def get_all_360_matches(self):
        """
        Retrieves a list of match IDs that are guaranteed to have 
        360 (freeze frame) data available.
        """
        print("Fetching competition data...")
        comps = sb.competitions()
        
        all_360_matches = []
        
        # We loop through all available competitions (World Cup, Euros, etc.)
        for _, row in comps.iterrows():
            try:
                # Fetch all matches for this specific competition and season
                matches = sb.matches(competition_id=row['competition_id'], 
                                    season_id=row['season_id'])
                
                # Check if the 360 status column exists (it isn't in older data)
                if 'match_status_360' in matches.columns:
                    # Filter for matches where 360 is 'available'
                    m360 = matches[matches['match_status_360'] == 'available']
                    match_ids = m360['match_id'].tolist()
                    all_360_matches.extend(match_ids)
                    
                    if len(match_ids) > 0:
                        print(f"Found {len(match_ids)} 360-enabled matches in {row['competition_name']} ({row['season_name']})")
                
            except Exception as e:
                # Some competitions might be restricted or empty; we skip them
                continue
                
        # Use set() to ensure we don't have duplicate IDs
        unique_360_matches = list(set(all_360_matches))
        print(f"\n--- LOADER DONE ---")
        print(f"Total 360-enabled matches found: {len(unique_360_matches)}")
        
        return unique_360_matches

# For standalone testing
if __name__ == "__main__":
    loader = StatsBombLoader()
    test_ids = loader.get_all_360_matches()
    print(f"First 5 IDs: {test_ids[:5]}")