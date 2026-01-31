
import pandas as pd
import numpy as np
import requests
from statsbombpy import sb

THREESIXTY_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{match_id}.json"

class MatchParser:
    def __init__(self, match_id, max_players=20):
    def __init__(self, match_id, max_players=20):
        self.match_id = match_id
        self.max_players = max_players
        
    def _fetch_360_data(self):
        """Fetch 360 data directly from StatsBomb GitHub and return as dict keyed by event_uuid."""
        url = THREESIXTY_URL.format(match_id=self.match_id)
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            data = response.json()
            return {item['event_uuid']: item for item in data}
        except Exception:
            return {}
        
    def parse(self):
        events = sb.events(match_id=self.match_id)
        
        # Fetch 360 data manually
        frames_lookup = self._fetch_360_data()
        
        cols_to_keep = [
            'id', 'index', 'period', 'timestamp', 'minute', 'second', 
            'type', 'location', 'pass_end_location', 'carry_end_location',
            'pass_outcome', 'shot_outcome', 'team_id'
        ]

        available_cols = [c for c in cols_to_keep if c in events.columns]
        df = events[available_cols].copy()

        # Filter to relevant event types
        df = df[df['type'].isin(['Pass', 'Carry', 'Shot'])].copy()
        df = df.reset_index(drop=True)
        
        # Extract start/end coordinates
        df['start_x'] = np.nan
        df['start_y'] = np.nan
        df['end_x'] = np.nan
        df['end_y'] = np.nan

        if 'location' in df.columns:
            loc_data = df['location'].dropna()
            if not loc_data.empty:
                coords = np.vstack(loc_data.values)
                df.loc[loc_data.index, 'start_x'] = coords[:, 0]
                df.loc[loc_data.index, 'start_y'] = coords[:, 1]

        if 'pass_end_location' in df.columns:
            pass_mask = (df['type'] == 'Pass') & (df['pass_end_location'].notna())
            if pass_mask.any():
                pass_coords = np.vstack(df.loc[pass_mask, 'pass_end_location'].values)
                df.loc[pass_mask, 'end_x'] = pass_coords[:, 0]
                df.loc[pass_mask, 'end_y'] = pass_coords[:, 1]

        if 'carry_end_location' in df.columns:
            carry_mask = (df['type'] == 'Carry') & (df['carry_end_location'].notna())
            if carry_mask.any():
                carry_coords = np.vstack(df.loc[carry_mask, 'carry_end_location'].values)
                df.loc[carry_mask, 'end_x'] = carry_coords[:, 0]
                df.loc[carry_mask, 'end_y'] = carry_coords[:, 1]

        # Determine success
        df['success'] = 0 
        if 'pass_outcome' in df.columns:
            df.loc[(df['type'] == 'Pass') & (df['pass_outcome'].isna()), 'success'] = 1
        df.loc[df['type'] == 'Carry', 'success'] = 1
        if 'shot_outcome' in df.columns:
            df.loc[(df['type'] == 'Shot') & (df['shot_outcome'] == 'Goal'), 'success'] = 1
            
        df['match_id'] = self.match_id
        
        # Sort by time
        df['timestamp_dt'] = pd.to_timedelta(df['timestamp'], errors='coerce')
        df = df.sort_values(by=['period', 'timestamp_dt', 'index']).reset_index(drop=True)

        # Extract player and keeper positions from 360 data
        player_features = df.apply(
            lambda row: self._extract_player_list(row, frames_lookup), 
            axis=1
        )
        player_df = pd.DataFrame(player_features.tolist(), index=df.index)
        
        # Build column names: players + keepers
        cols = []
        for i in range(self.max_players):
            cols.extend([f'p{i}_x', f'p{i}_y', f'p{i}_team'])
        cols.extend(['keeper1_x', 'keeper1_y', 'keeper1_team', 'keeper2_x', 'keeper2_y', 'keeper2_team'])
        player_df.columns = cols
        
        result_cols = ['match_id', 'index', 'period', 'timestamp', 'minute', 'second', 'type', 
                       'team_id', 'start_x', 'start_y', 'end_x', 'end_y', 'success']
        
        return pd.concat([df[result_cols], player_df], axis=1)
    
    def _extract_player_list(self, row, frames_lookup):
        """Extract player positions from 360 freeze frame data."""
        event_id = row.get('id')
        frame_data = frames_lookup.get(event_id, {})
        freeze_frame = frame_data.get('freeze_frame', [])
        
        output = []
        
        if freeze_frame and len(freeze_frame) > 0:
            # Separate keepers from outfield players
            keepers = []
            outfield = []
            
            for p in freeze_frame:
                if p.get('keeper', False):
                    keepers.append(p)
                else:
                    # Skip the actor (ball carrier) - they're at ball position
                    if not p.get('actor', False):
                        outfield.append(p)
            
            # Sort outfield players by proximity to the ball
            ball_x, ball_y = row['start_x'], row['start_y']
            if pd.notna(ball_x) and pd.notna(ball_y):
                outfield = sorted(outfield, key=lambda p: np.sqrt(
                    (p['location'][0] - ball_x)**2 + (p['location'][1] - ball_y)**2
                ))
            
            # Extract outfield player positions
            for i in range(self.max_players):
                if i < len(outfield):
                    p = outfield[i]
                    output.extend([
                        p['location'][0], 
                        p['location'][1], 
                        1 if p.get('teammate', False) else 0
                    ])
                else:
                    output.extend([None, None, None])
            
            # Extract keeper positions (up to 2 keepers)
            for i in range(2):
                if i < len(keepers):
                    k = keepers[i]
                    output.extend([
                        k['location'][0], 
                        k['location'][1], 
                        1 if k.get('teammate', False) else 0
                    ])
                else:
                    output.extend([None, None, None])
        else:
            # No 360 data for this event
            output = [None] * (self.max_players * 3 + 6)  # +6 for 2 keepers
            
        return output