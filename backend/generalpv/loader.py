from statsbombpy import sb
import warnings

warnings.filterwarnings('ignore')

class StatsBombLoader:
    """
    Responsible for fetching metadata about competitions and matches.
    """

    
    def get_all_free_matches(self):
        print("Fetching competition data...")
        comps = sb.competitions()
        
        all_matches = []
        print(f"Found {len(comps)} competitions. Fetching matches...")
        
        for index, row in comps.iterrows():
            try:
                matches = sb.matches(competition_id=row['competition_id'], season_id=row['season_id'])
                match_ids = matches['match_id'].tolist()
                all_matches.extend(match_ids)
            except Exception as e:
                continue
                
        print(f"Total matches found: {len(all_matches)}")
        return list(set(all_matches))

    def get_all_360_matches(self):
        print("Fetching competition data...")
        comps = sb.competitions()
        
        # StatsBomb 360 is generally available for Euro 2020, 
        # World Cup 2022, and recent top-flight seasons.
        all_360_matches = []
        
        for _, row in comps.iterrows():
            try:
                matches = sb.matches(competition_id=row['competition_id'], 
                                     season_id=row['season_id'])
                
                # Check if the match has the 'match_status_360' column 
                # and if it's 'available'
                if 'match_status_360' in matches.columns:
                    m360 = matches[matches['match_status_360'] == 'available']
                    all_360_matches.extend(m360['match_id'].tolist())
            except Exception:
                continue
                
        print(f"Total 360-enabled matches found: {len(all_360_matches)}")
        return list(set(all_360_matches))
    
loader = StatsBombLoader()
loader.get_all_360_matches()