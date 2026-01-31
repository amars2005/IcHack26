
import pandas as pd
import numpy as np
from statsbombpy import sb

class MatchParser:
    def __init__(self, match_id, max_players=20):
        self.match_id = match_id
        self.max_players = max_players
        
    def parse(self):
        events = sb.events(match_id=self.match_id)
        frames = sb.frames(match_id=self.match_id)
        cols_to_keep = [
            'id', 'index', 'period', 'timestamp', 'minute', 'second', 
            'type', 'location', 'pass_end_location', 'carry_end_location',
            'pass_outcome', 'shot_outcome', 'team_id'
        ]
        

        available_cols = [c for c in cols_to_keep if c in events.columns]
        df = events[available_cols].copy()

        if not frames.empty:
            df = df.merge(frames, on='id', how='left')
        else:
            # Create an empty freeze_frame column if no 360 data exists for this match
            df['freeze_frame'] = np.nan
        
        df = df[df['type'].isin(['Pass', 'Carry', 'Shot'])].copy()
        
        df['start_x'] = np.nan; df['start_y'] = np.nan
        df['end_x'] = np.nan; df['end_y'] = np.nan

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

        df['success'] = 0 
        if 'pass_outcome' in df.columns:
            df.loc[(df['type'] == 'Pass') & (df['pass_outcome'].isna()), 'success'] = 1
        df.loc[df['type'] == 'Carry', 'success'] = 1
        if 'shot_outcome' in df.columns:
            df.loc[(df['type'] == 'Shot') & (df['shot_outcome'] == 'Goal'), 'success'] = 1
            
        df['match_id'] = self.match_id
        
        df['timestamp_dt'] = pd.to_timedelta(df['timestamp'], errors='coerce')
        df = df.sort_values(by=['period', 'timestamp_dt', 'index'])

        player_features = df.apply(self._extract_player_list, axis=1)
        player_df = pd.DataFrame(player_features.tolist(), index=df.index)
        
        cols = []
        for i in range(self.max_players):
            cols.extend([f'p{i}_x', f'p{i}_y', f'p{i}_team'])
        player_df.columns = cols
        
        return pd.concat([df[['match_id', 'index', 'period', 'timestamp', 'minute', 'second', 'type', 
                   'team_id', 'start_x', 'start_y', 'end_x', 'end_y', 'success']], player_df], axis=1)
    def _extract_player_list(self, row):
        frame = row.get('freeze_frame')
        output = []
        
        if isinstance(frame, list):
            # Sort players by proximity to the ball so the model sees the most relevant players first
            sorted_frame = sorted(frame, key=lambda p: np.sqrt((p['location'][0]-row['start_x'])**2 + 
                                                               (p['location'][1]-row['start_y'])**2))
            
            for i in range(self.max_players):
                if i < len(sorted_frame):
                    p = sorted_frame[i]
                    output.extend([p['location'][0], p['location'][1], 1 if p['teammate'] else 0])
                else:
                    # Padding: Fill empty slots with -1
                    output.extend([-1, -1, -1])
        else:
            # No 360 data for this event
            output = [-1] * (self.max_players * 3)
            
        return output

